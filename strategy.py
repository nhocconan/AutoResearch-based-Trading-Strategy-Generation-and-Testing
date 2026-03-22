#!/usr/bin/env python3
"""
Experiment #538: 30m Primary + 4h/1d HTF — HMA Trend + Connors RSI + Volume Filter

Hypothesis: After 480+ failed strategies, lower TF (30m) needs STRICT confluence
to avoid fee drag while maintaining trade frequency. Key insight from #536 success:
simple trend-following with pullback entries works better than complex regime logic.

Strategy design for 30m (target 30-80 trades/year across all symbols):
1. 1d HMA(21) for MAJOR trend direction (primary filter)
2. 4h HMA(21) for INTERMEDIATE trend confirmation (secondary filter)
3. 30m Connors RSI for pullback entries (RSI(3) + RSI_Streak(2) + PercentRank(100)) / 3
4. ADX(14) > 18 to avoid choppy whipsaws (slightly lower than 12h strategy)
5. Volume > 0.8x 20-bar average (confirm participation)
6. ATR(14) 2.5x trailing stop for risk management

Why 30m can work:
- HTF (1d/4h) provides trend direction = fewer counter-trend trades
- Connors RSI catches pullbacks within trend = better entry timing
- Volume filter avoids low-liquidity false breakouts
- ADX filter reduces choppy market whipsaws
- Discrete position sizing (0.22) minimizes fee churn on signal changes

Position sizing: 0.22 (lower than 12h due to more frequent signals)
Stoploss: 2.5 * ATR trailing (signal → 0 when hit)
Target: >=40 trades/symbol on train (4 years), >=3 on test, Sharpe > 0 all symbols
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_30m_hma_connors_rsi_volume_4h1d_v1"
timeframe = "30m"
leverage = 1.0

def calculate_atr(high, low, close, period=14):
    """Calculate ATR using Wilder's smoothing."""
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]
    atr = pd.Series(tr).ewm(span=period, min_periods=period, adjust=False).mean().values
    return atr

def calculate_rsi(close, period=14):
    """Calculate RSI using Wilder's smoothing."""
    close_s = pd.Series(close)
    delta = close_s.diff()
    
    gain = delta.where(delta > 0, 0.0)
    loss = -delta.where(delta < 0, 0.0)
    
    avg_gain = gain.ewm(span=period, min_periods=period, adjust=False).mean()
    avg_loss = loss.ewm(span=period, min_periods=period, adjust=False).mean()
    
    rs = avg_gain / (avg_loss + 1e-10)
    rsi = 100.0 - (100.0 / (1.0 + rs))
    
    return rsi.values

def calculate_rsi_streak(close, period=2):
    """
    Calculate RSI Streak component of Connors RSI.
    Measures consecutive up/down days.
    """
    n = len(close)
    streak = np.zeros(n)
    
    for i in range(1, n):
        if close[i] > close[i-1]:
            streak[i] = streak[i-1] + 1 if streak[i-1] >= 0 else 1
        elif close[i] < close[i-1]:
            streak[i] = streak[i-1] - 1 if streak[i-1] <= 0 else -1
        else:
            streak[i] = 0
    
    # Convert streak to RSI-like value (0-100)
    # Positive streak = bullish, negative = bearish
    streak_rsi = np.zeros(n)
    for i in range(period, n):
        lookback = streak[max(0, i-period+1):i+1]
        up_count = np.sum(lookback > 0)
        streak_rsi[i] = (up_count / period) * 100.0
    
    return streak_rsi

def calculate_percent_rank(close, period=100):
    """
    Calculate Percent Rank component of Connors RSI.
    Measures where current return ranks vs last period returns.
    """
    n = len(close)
    pct_rank = np.zeros(n)
    
    returns = np.zeros(n)
    returns[1:] = (close[1:] - close[:-1]) / (close[:-1] + 1e-10) * 100.0
    
    for i in range(period, n):
        lookback = returns[max(0, i-period+1):i]
        if len(lookback) > 0:
            current_return = returns[i]
            pct_rank[i] = (np.sum(lookback < current_return) / len(lookback)) * 100.0
    
    return pct_rank

def calculate_connors_rsi(close, rsi_period=3, streak_period=2, pr_period=100):
    """
    Calculate Connors RSI (CRSI).
    CRSI = (RSI(3) + RSI_Streak(2) + PercentRank(100)) / 3
    Values 0-100. <10 = oversold, >90 = overbought.
    """
    rsi_3 = calculate_rsi(close, rsi_period)
    streak_rsi = calculate_rsi_streak(close, streak_period)
    pct_rank = calculate_percent_rank(close, pr_period)
    
    crsi = (rsi_3 + streak_rsi + pct_rank) / 3.0
    return crsi

def calculate_adx(high, low, close, period=14):
    """Calculate ADX (Average Directional Index) for trend strength."""
    n = len(close)
    
    high_s = pd.Series(high)
    low_s = pd.Series(low)
    
    plus_dm = high_s.diff()
    minus_dm = -low_s.diff()
    
    plus_dm[plus_dm < 0] = 0
    minus_dm[minus_dm < 0] = 0
    
    mask = (plus_dm > 0) & (minus_dm > 0)
    plus_dm_vals = plus_dm.values.copy()
    minus_dm_vals = minus_dm.values.copy()
    plus_dm_vals[mask] = np.where(plus_dm_vals[mask] > minus_dm_vals[mask], plus_dm_vals[mask], 0)
    minus_dm_vals[mask] = np.where(minus_dm_vals[mask] > plus_dm_vals[mask], minus_dm_vals[mask], 0)
    
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]
    
    plus_dm_s = pd.Series(plus_dm_vals).ewm(span=period, min_periods=period, adjust=False).mean().values
    minus_dm_s = pd.Series(minus_dm_vals).ewm(span=period, min_periods=period, adjust=False).mean().values
    tr_s = pd.Series(tr).ewm(span=period, min_periods=period, adjust=False).mean().values
    
    plus_di = 100.0 * plus_dm_s / (tr_s + 1e-10)
    minus_di = 100.0 * minus_dm_s / (tr_s + 1e-10)
    
    dx = 100.0 * np.abs(plus_di - minus_di) / (plus_di + minus_di + 1e-10)
    adx = pd.Series(dx).ewm(span=period, min_periods=period, adjust=False).mean().values
    
    return adx

def calculate_hma(close, period=21):
    """Calculate Hull Moving Average (HMA) - reduces lag vs EMA."""
    n = period
    half = n // 2
    sqrt_n = int(np.sqrt(n))
    
    close_s = pd.Series(close)
    
    def wma(series, span):
        weights = np.arange(1, span + 1)
        return series.rolling(window=span, min_periods=span).apply(
            lambda x: np.dot(x, weights) / weights.sum(), raw=True
        )
    
    wma_half = wma(close_s, half)
    wma_full = wma(close_s, n)
    hma_raw = 2.0 * wma_half - wma_full
    hma = wma(hma_raw, sqrt_n)
    
    return hma.values

def generate_signals(prices):
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    volume = prices["volume"].values
    n = len(close)
    
    # Load HTF data ONCE before loop (Rule 1 - CRITICAL)
    df_4h = get_htf_data(prices, '4h')
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate HTF HMAs
    hma_1d_21 = calculate_hma(df_1d['close'].values, period=21)
    hma_4h_21 = calculate_hma(df_4h['close'].values, period=21)
    hma_4h_50 = calculate_hma(df_4h['close'].values, period=50)
    
    # Align HTF to LTF (Rule 2 - auto shift(1) for completed bars)
    hma_1d_21_aligned = align_htf_to_ltf(prices, df_1d, hma_1d_21)
    hma_4h_21_aligned = align_htf_to_ltf(prices, df_4h, hma_4h_21)
    hma_4h_50_aligned = align_htf_to_ltf(prices, df_4h, hma_4h_50)
    
    # Calculate 30m indicators
    atr_14 = calculate_atr(high, low, close, 14)
    adx_14 = calculate_adx(high, low, close, 14)
    crsi = calculate_connors_rsi(close, rsi_period=3, streak_period=2, pr_period=100)
    
    # Volume average (20 bars)
    vol_avg_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    
    # Position sizing (Rule 4 - discrete, lower for 30m due to more signals)
    POSITION_SIZE = 0.22
    
    # Track position state for stoploss
    in_position = False
    position_side = 0
    entry_price = 0.0
    highest_since_entry = 0.0
    lowest_since_entry = 0.0
    
    for i in range(150, n):
        # Skip if indicators not ready
        if np.isnan(atr_14[i]) or atr_14[i] == 0:
            signals[i] = signals[i-1] if i > 0 else 0.0
            continue
        if np.isnan(hma_1d_21_aligned[i]) or np.isnan(hma_4h_21_aligned[i]):
            signals[i] = signals[i-1] if i > 0 else 0.0
            continue
        if np.isnan(crsi[i]) or np.isnan(adx_14[i]) or np.isnan(vol_avg_20[i]):
            signals[i] = signals[i-1] if i > 0 else 0.0
            continue
        
        # === 1D MAJOR TREND (primary direction filter) ===
        bull_regime_1d = close[i] > hma_1d_21_aligned[i]
        bear_regime_1d = close[i] < hma_1d_21_aligned[i]
        
        # === 4H INTERMEDIATE TREND (confirmation filter) ===
        bull_regime_4h = close[i] > hma_4h_21_aligned[i]
        bear_regime_4h = close[i] < hma_4h_21_aligned[i]
        
        # 4h HMA slope for trend strength
        hma_4h_slope_bull = hma_4h_21_aligned[i] > hma_4h_50_aligned[i]
        hma_4h_slope_bear = hma_4h_21_aligned[i] < hma_4h_50_aligned[i]
        
        # === ADX FILTER (trending market) ===
        trending = adx_14[i] > 18.0  # Market has some trend
        
        # === CONNORS RSI EXTREMES (pullback entries) ===
        crsi_oversold = crsi[i] < 20.0  # Strong oversold for long
        crsi_overbought = crsi[i] > 80.0  # Strong overbought for short
        crsi_extreme_long = crsi[i] < 15.0  # Very strong oversold
        crsi_extreme_short = crsi[i] > 85.0  # Very strong overbought
        
        # === VOLUME FILTER ===
        volume_ok = volume[i] > 0.8 * vol_avg_20[i]
        volume_strong = volume[i] > 1.2 * vol_avg_20[i]
        
        # === ENTRY LOGIC — STRICT CONFLUENCE FOR LOWER TF ===
        new_signal = 0.0
        
        # LONG ENTRIES (need 3+ confluence)
        # Condition 1: 1d bull + 4h bull + ADX trending + CRSI oversold + volume
        if bull_regime_1d and bull_regime_4h and trending and crsi_oversold and volume_ok:
            new_signal = POSITION_SIZE
        # Condition 2: 1d bull + 4h bull slope + CRSI extreme + volume strong
        elif bull_regime_1d and hma_4h_slope_bull and crsi_extreme_long and volume_strong:
            new_signal = POSITION_SIZE
        # Condition 3: 1d bull + 4h bull + CRSI very oversold (deep pullback)
        elif bull_regime_1d and bull_regime_4h and crsi[i] < 12.0:
            new_signal = POSITION_SIZE * 0.8
        # Condition 4: All bull alignment + CRSI recovering from extreme
        elif bull_regime_1d and bull_regime_4h and hma_4h_slope_bull and crsi[i] < 25.0 and crsi[i] > crsi[i-1]:
            new_signal = POSITION_SIZE * 0.8
        
        # SHORT ENTRIES (mirror logic)
        if new_signal == 0.0:
            # Condition 1: 1d bear + 4h bear + ADX trending + CRSI overbought + volume
            if bear_regime_1d and bear_regime_4h and trending and crsi_overbought and volume_ok:
                new_signal = -POSITION_SIZE
            # Condition 2: 1d bear + 4h bear slope + CRSI extreme + volume strong
            elif bear_regime_1d and hma_4h_slope_bear and crsi_extreme_short and volume_strong:
                new_signal = -POSITION_SIZE
            # Condition 3: 1d bear + 4h bear + CRSI very overbought (deep bounce)
            elif bear_regime_1d and bear_regime_4h and crsi[i] > 88.0:
                new_signal = -POSITION_SIZE * 0.8
            # Condition 4: All bear alignment + CRSI rolling over from extreme
            elif bear_regime_1d and bear_regime_4h and hma_4h_slope_bear and crsi[i] > 75.0 and crsi[i] < crsi[i-1]:
                new_signal = -POSITION_SIZE * 0.8
        
        # === HOLD POSITION LOGIC ===
        # If already in position, maintain unless exit conditions hit
        if in_position and new_signal == 0.0:
            new_signal = signals[i-1] if i > 0 else 0.0
        
        # === STOPLOSS CHECK (2.5 * ATR trailing) ===
        stoploss_triggered = False
        
        if in_position and position_side > 0:
            highest_since_entry = max(highest_since_entry, close[i])
            stop_price = highest_since_entry - 2.5 * atr_14[i]
            if close[i] < stop_price:
                stoploss_triggered = True
        
        if in_position and position_side < 0:
            if lowest_since_entry == 0.0:
                lowest_since_entry = close[i]
            else:
                lowest_since_entry = min(lowest_since_entry, close[i])
            stop_price = lowest_since_entry + 2.5 * atr_14[i]
            if close[i] > stop_price:
                stoploss_triggered = True
        
        if stoploss_triggered:
            new_signal = 0.0
        
        # === EXIT CONDITIONS (regime flip or CRSI extreme reversal) ===
        # Exit long on regime flip to bear
        if in_position and position_side > 0:
            if bear_regime_1d and bear_regime_4h:
                new_signal = 0.0
            elif crsi[i] > 85.0:  # CRSI very overbought = take profit
                new_signal = 0.0
        
        # Exit short on regime flip to bull
        if in_position and position_side < 0:
            if bull_regime_1d and bull_regime_4h:
                new_signal = 0.0
            elif crsi[i] < 15.0:  # CRSI very oversold = take profit
                new_signal = 0.0
        
        # === UPDATE POSITION TRACKING ===
        if new_signal != 0.0:
            if not in_position:
                in_position = True
                position_side = np.sign(new_signal)
                entry_price = close[i]
                highest_since_entry = close[i] if position_side > 0 else 0.0
                lowest_since_entry = close[i] if position_side < 0 else 0.0
            elif np.sign(new_signal) != position_side:
                # Flip position
                position_side = np.sign(new_signal)
                entry_price = close[i]
                highest_since_entry = close[i] if position_side > 0 else 0.0
                lowest_since_entry = close[i] if position_side < 0 else 0.0
        else:
            if in_position:
                in_position = False
                position_side = 0
                entry_price = 0.0
                highest_since_entry = 0.0
                lowest_since_entry = 0.0
        
        signals[i] = new_signal
    
    return signals