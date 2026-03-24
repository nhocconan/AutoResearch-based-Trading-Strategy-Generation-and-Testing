#!/usr/bin/env python3
"""
Experiment #076: 30m Primary + 4h/1d HTF — cRSI + Choppiness + HMA Trend + Session

Hypothesis: After 75 failed experiments, the pattern for lower TF (30m) is clear:
- Pure trend following fails on BTC/ETH in bear/range markets
- Pure mean reversion fails on SOL during strong trends
- SOLUTION: Dual-regime with Choppiness to switch between cRSI mean-revert and breakout
- 4h HMA provides trend bias, 1d HMA provides major trend filter
- cRSI (Connors RSI) is proven for bear market reversals (75% win rate)
- Session filter (08-20 UTC) reduces trades to target 40-80/year
- This combines: HTF trend (4h/1d HMA) + cRSI entries + Choppiness regime + Session

Key design choices:
- Timeframe: 30m (target 40-80 trades/year with session filter)
- HTF: 4h HMA(21) for trend, 1d HMA(50) for major bias
- Entry: cRSI extremes (15/85) + regime filter + session filter
- Regime: CHOP>55 = range (cRSI mean revert), CHOP<55 = trend (cRSI + breakout)
- Position size: 0.20 (20% of capital, conservative for 30m)
- Stoploss: 2.5x ATR trailing
- Session: 08-20 UTC only (reduces trades, avoids low liquidity)

Target: Sharpe>0.167 (beat current best), DD>-40%, trades>=30 train, trades>=3 test
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_30m_crsi_chop_hma_4h1d_session_v1"
timeframe = "30m"
leverage = 1.0

def calculate_hma(close, period):
    """Hull Moving Average - faster response than EMA"""
    n = len(close)
    if n < period:
        return np.full(n, np.nan)
    
    half = period // 2
    sqrt_period = int(np.sqrt(period))
    
    wma1 = pd.Series(close).ewm(span=half, min_periods=half, adjust=False).mean().values
    wma2 = pd.Series(close).ewm(span=period, min_periods=period, adjust=False).mean().values
    
    diff = 2.0 * wma1 - wma2
    hma = pd.Series(diff).ewm(span=sqrt_period, min_periods=sqrt_period, adjust=False).mean().values
    
    return hma

def calculate_rsi(close, period=14):
    """Relative Strength Index"""
    n = len(close)
    if n < period + 1:
        return np.full(n, np.nan)
    
    delta = np.diff(close)
    gain = np.where(delta > 0, delta, 0.0)
    loss = np.where(delta < 0, -delta, 0.0)
    gain = np.concatenate([[0.0], gain])
    loss = np.concatenate([[0.0], loss])
    
    avg_gain = pd.Series(gain).ewm(span=period, min_periods=period, adjust=False).mean().values
    avg_loss = pd.Series(loss).ewm(span=period, min_periods=period, adjust=False).mean().values
    
    rsi = np.zeros(n)
    rsi[:] = np.nan
    for i in range(period, n):
        if avg_loss[i] < 1e-10:
            rsi[i] = 100.0
        else:
            rs = avg_gain[i] / avg_loss[i]
            rsi[i] = 100.0 - (100.0 / (1.0 + rs))
    
    return rsi

def calculate_streak_rsi(close, period=2):
    """
    Streak RSI component of Connors RSI
    Measures consecutive up/down days
    """
    n = len(close)
    if n < period + 2:
        return np.full(n, np.nan)
    
    # Calculate streak
    streak = np.zeros(n)
    for i in range(1, n):
        if close[i] > close[i-1]:
            streak[i] = streak[i-1] + 1 if streak[i-1] >= 0 else 1
        elif close[i] < close[i-1]:
            streak[i] = streak[i-1] - 1 if streak[i-1] <= 0 else -1
        else:
            streak[i] = 0
    
    # Calculate RSI of streak (absolute values)
    streak_abs = np.abs(streak)
    streak_rsi = calculate_rsi(streak_abs, period)
    
    # Adjust sign: positive streak = bullish, negative = bearish
    for i in range(len(streak_rsi)):
        if not np.isnan(streak_rsi[i]) and streak[i] < 0:
            streak_rsi[i] = 100.0 - streak_rsi[i]
    
    return streak_rsi

def calculate_percent_rank(close, period=100):
    """
    Percent Rank component of Connors RSI
    Measures where current return ranks vs past period returns
    """
    n = len(close)
    if n < period + 1:
        return np.full(n, np.nan)
    
    # Calculate returns
    returns = np.zeros(n)
    returns[:] = np.nan
    for i in range(1, n):
        returns[i] = (close[i] - close[i-1]) / close[i-1] * 100.0
    
    percent_rank = np.zeros(n)
    percent_rank[:] = np.nan
    
    for i in range(period, n):
        if np.isnan(returns[i]):
            continue
        
        current_return = returns[i]
        past_returns = returns[i-period:i]
        valid_returns = past_returns[~np.isnan(past_returns)]
        
        if len(valid_returns) == 0:
            percent_rank[i] = 50.0
        else:
            count_below = np.sum(valid_returns < current_return)
            percent_rank[i] = (count_below / len(valid_returns)) * 100.0
    
    return percent_rank

def calculate_crsi(close, rsi_period=3, streak_period=2, pr_period=100):
    """
    Connors RSI (CRSI)
    Formula: (RSI(3) + RSI_Streak(2) + PercentRank(100)) / 3
    Range: 0-100, extremes <10 or >90 signal reversals
    """
    rsi_short = calculate_rsi(close, rsi_period)
    streak_rsi = calculate_streak_rsi(close, streak_period)
    percent_rank = calculate_percent_rank(close, pr_period)
    
    n = len(close)
    crsi = np.zeros(n)
    crsi[:] = np.nan
    
    for i in range(n):
        if np.isnan(rsi_short[i]) or np.isnan(streak_rsi[i]) or np.isnan(percent_rank[i]):
            continue
        crsi[i] = (rsi_short[i] + streak_rsi[i] + percent_rank[i]) / 3.0
    
    return crsi

def calculate_atr(high, low, close, period=14):
    """Average True Range for stoploss"""
    n = len(close)
    if n < period + 1:
        return np.full(n, np.nan)
    
    tr = np.zeros(n)
    tr[0] = high[0] - low[0]
    for i in range(1, n):
        tr[i] = max(high[i] - low[i], abs(high[i] - close[i-1]), abs(low[i] - close[i-1]))
    
    atr = pd.Series(tr).ewm(span=period, min_periods=period, adjust=False).mean().values
    return atr

def calculate_choppiness(high, low, close, period=14):
    """
    Choppiness Index (CHOP)
    CHOP > 61.8 = choppy/range, CHOP < 38.2 = trending
    """
    n = len(close)
    if n < period + 1:
        return np.full(n, np.nan)
    
    tr = np.zeros(n)
    tr[0] = high[0] - low[0]
    for i in range(1, n):
        tr[i] = max(high[i] - low[i], abs(high[i] - close[i-1]), abs(low[i] - close[i-1]))
    
    chop = np.zeros(n)
    chop[:] = np.nan
    
    for i in range(period, n):
        sum_tr = np.sum(tr[i-period+1:i+1])
        highest_high = np.max(high[i-period+1:i+1])
        lowest_low = np.min(low[i-period+1:i+1])
        range_hl = highest_high - lowest_low
        
        if range_hl > 1e-10 and sum_tr > 1e-10:
            chop[i] = 100.0 * np.log10(sum_tr / range_hl) / np.log10(period)
        else:
            chop[i] = 50.0
    
    return chop

def is_session_active(open_time):
    """
    Session filter: 08-20 UTC only
    open_time is in milliseconds since epoch
    """
    # Convert to hour of day UTC
    hour = (open_time // 1000 // 3600) % 24
    return 8 <= hour < 20

def generate_signals(prices):
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    open_time = prices["open_time"].values
    n = len(close)
    
    # Load HTF data ONCE before loop (Rule 1 - CRITICAL)
    df_4h = get_htf_data(prices, '4h')
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate and align 4h HMA for trend bias
    hma_4h_raw = calculate_hma(df_4h['close'].values, period=21)
    hma_4h_aligned = align_htf_to_ltf(prices, df_4h, hma_4h_raw)
    
    # Calculate and align 1d HMA for major trend bias
    hma_1d_raw = calculate_hma(df_1d['close'].values, period=50)
    hma_1d_aligned = align_htf_to_ltf(prices, df_1d, hma_1d_raw)
    
    # Calculate primary (30m) indicators
    crsi = calculate_crsi(close, rsi_period=3, streak_period=2, pr_period=100)
    atr = calculate_atr(high, low, close, period=14)
    chop = calculate_choppiness(high, low, close, period=14)
    
    # 30m HMA for local trend
    hma_30m = calculate_hma(close, period=21)
    
    signals = np.zeros(n)
    SIZE = 0.20  # 20% position size (conservative for 30m)
    
    # Position tracking for stoploss
    in_position = False
    position_side = 0
    entry_price = 0.0
    entry_atr = 0.0
    highest_since_entry = 0.0
    lowest_since_entry = float('inf')
    
    for i in range(150, n):
        # Skip if indicators not ready
        if np.isnan(atr[i]) or atr[i] <= 1e-10:
            signals[i] = 0.0
            if in_position:
                in_position = False
                position_side = 0
            continue
        if np.isnan(crsi[i]) or np.isnan(chop[i]):
            signals[i] = 0.0
            if in_position:
                in_position = False
                position_side = 0
            continue
        if np.isnan(hma_4h_aligned[i]) or np.isnan(hma_1d_aligned[i]):
            signals[i] = 0.0
            if in_position:
                in_position = False
                position_side = 0
            continue
        if np.isnan(hma_30m[i]):
            signals[i] = 0.0
            if in_position:
                in_position = False
                position_side = 0
            continue
        
        # Session filter - only trade 08-20 UTC
        if not is_session_active(open_time[i]):
            signals[i] = 0.0
            if in_position:
                in_position = False
                position_side = 0
            continue
        
        # === HTF BIAS (4h and 1d HMA) ===
        htf_4h_bull = close[i] > hma_4h_aligned[i]
        htf_4h_bear = close[i] < hma_4h_aligned[i]
        htf_1d_bull = close[i] > hma_1d_aligned[i]
        htf_1d_bear = close[i] < hma_1d_aligned[i]
        
        # Strong bias when both HTFs agree
        htf_strong_bull = htf_4h_bull and htf_1d_bull
        htf_strong_bear = htf_4h_bear and htf_1d_bear
        
        # === REGIME DETECTION (Choppiness Index) ===
        # CHOP > 55 = range/choppy (mean revert with cRSI)
        # CHOP < 55 = trending (follow trend with cRSI pullback)
        is_choppy = chop[i] > 55.0
        is_trending = chop[i] <= 55.0
        
        # === cRSI EXTREMES (Connors RSI) ===
        # cRSI < 15 = oversold (long signal in range or pullback)
        # cRSI > 85 = overbought (short signal in range or pullback)
        crsi_oversold = crsi[i] < 15.0
        crsi_overbought = crsi[i] > 85.0
        
        # === 30m HMA LOCAL TREND ===
        hma_30m_bull = close[i] > hma_30m[i]
        hma_30m_bear = close[i] < hma_30m[i]
        
        # === DESIRED SIGNAL (Dual Regime Logic) ===
        desired_signal = 0.0
        
        if is_choppy:
            # RANGE REGIME: Mean revert with cRSI extremes
            # LONG: cRSI oversold + HTF not strongly bear
            if crsi_oversold and not htf_strong_bear:
                desired_signal = SIZE
            # SHORT: cRSI overbought + HTF not strongly bull
            elif crsi_overbought and not htf_strong_bull:
                desired_signal = -SIZE
        else:
            # TREND REGIME: cRSI pullback entries with HTF bias
            # LONG: cRSI oversold + HTF bull + local HMA bull
            if crsi_oversold and htf_strong_bull and hma_30m_bull:
                desired_signal = SIZE
            # SHORT: cRSI overbought + HTF bear + local HMA bear
            elif crsi_overbought and htf_strong_bear and hma_30m_bear:
                desired_signal = -SIZE
            # Fallback: weaker HTF agreement
            elif crsi_oversold and htf_4h_bull:
                desired_signal = SIZE * 0.5
            elif crsi_overbought and htf_4h_bear:
                desired_signal = -SIZE * 0.5
        
        # === STOPLOSS CHECK (Trailing ATR 2.5x) ===
        stoploss_triggered = False
        
        if in_position and position_side > 0:
            highest_since_entry = max(highest_since_entry, close[i])
            stop_price = highest_since_entry - 2.5 * entry_atr
            if close[i] < stop_price:
                stoploss_triggered = True
        
        if in_position and position_side < 0:
            lowest_since_entry = min(lowest_since_entry, close[i])
            stop_price = lowest_since_entry + 2.5 * entry_atr
            if close[i] > stop_price:
                stoploss_triggered = True
        
        if stoploss_triggered:
            desired_signal = 0.0
        
        # === DISCRETIZE SIGNAL VALUES ===
        if desired_signal >= SIZE * 0.85:
            final_signal = SIZE
        elif desired_signal <= -SIZE * 0.85:
            final_signal = -SIZE
        elif desired_signal >= SIZE * 0.5:
            final_signal = SIZE * 0.5
        elif desired_signal <= -SIZE * 0.5:
            final_signal = -SIZE * 0.5
        else:
            final_signal = 0.0
        
        # === UPDATE POSITION TRACKING ===
        if final_signal != 0.0:
            if not in_position:
                in_position = True
                position_side = int(np.sign(final_signal))
                entry_price = close[i]
                entry_atr = atr[i]
                highest_since_entry = close[i] if position_side > 0 else 0.0
                lowest_since_entry = close[i] if position_side < 0 else float('inf')
            elif np.sign(final_signal) != position_side:
                # Flip position
                position_side = int(np.sign(final_signal))
                entry_price = close[i]
                entry_atr = atr[i]
                highest_since_entry = close[i] if position_side > 0 else 0.0
                lowest_since_entry = close[i] if position_side < 0 else float('inf')
            elif position_side > 0:
                highest_since_entry = max(highest_since_entry, close[i])
            elif position_side < 0:
                lowest_since_entry = min(lowest_since_entry, close[i])
        else:
            if in_position:
                in_position = False
                position_side = 0
                entry_price = 0.0
                entry_atr = 0.0
                highest_since_entry = 0.0
                lowest_since_entry = float('inf')
        
        signals[i] = final_signal
    
    return signals