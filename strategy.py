#!/usr/bin/env python3
"""
Experiment #777: 1d Primary + 1w HTF — Choppiness Regime + Connors RSI + Donchian Breakout

Hypothesis: After analyzing 500+ failed strategies and current best (Sharpe=0.612):
1. 1d timeframe generates fewer trades (20-50/year) = less fee drag, higher quality signals
2. 1w HMA provides cleaner trend bias than 1d/4h for daily entries (less noise)
3. Choppiness Index (14) > 61.8 = range regime (mean revert with Connors RSI)
4. Choppiness Index < 38.2 = trend regime (breakout with Donchian)
5. Connors RSI < 15 + price > weekly HMA = high-probability long in range
6. Donchian(20) breakout + weekly trend = trend continuation entry
7. ATR(14) 2.5x trailing stop protects capital in crypto volatility
8. Discrete signals (0.0, ±0.25, ±0.30) minimize fee churn from signal changes

Strategy design:
1. 1w HMA(21) for primary trend bias (aligned via mtf_data helper)
2. 1d Choppiness Index(14) for regime detection (range vs trend)
3. 1d Connors RSI for mean reversion entries in range regime
4. 1d Donchian(20) for breakout entries in trend regime
5. 1d ATR(14) for trailing stoploss (2.5x)
6. Position sizing: 0.25-0.30 (conservative for drawdown control)

Key improvements from failed experiments:
- Simpler regime logic (binary: chop vs trend, not triple regime)
- Fewer conflicting filters (removed volume, session, ADX hysteresis)
- Clearer entry conditions (CRSI extreme + trend OR Donchian breakout)
- 1d timeframe = naturally fewer trades, less fee drag
- 1w HTF = more stable trend filter than 12h/4h for daily bars

Target: Sharpe > 0.612, trades >= 30 train, >= 3 test, ALL symbols positive
Timeframe: 1d (target 20-50 trades/year)
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_1d_chop_regime_crsi_donchian_1w_hma_atr_v1"
timeframe = "1d"
leverage = 1.0

def calculate_hma(series, period):
    """Hull Moving Average - faster response than EMA."""
    series = pd.Series(series)
    wma1 = series.ewm(span=period // 2, min_periods=period // 2, adjust=False).mean()
    wma2 = series.ewm(span=period, min_periods=period, adjust=False).mean()
    hull = 2 * wma1 - wma2
    hma = hull.ewm(span=int(np.sqrt(period)), min_periods=int(np.sqrt(period)), adjust=False).mean()
    return hma.values

def calculate_rsi(close, period=14):
    """Relative Strength Index."""
    n = len(close)
    rsi = np.full(n, np.nan)
    
    if n < period + 1:
        return rsi
    
    delta = np.diff(close)
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    
    avg_gain = pd.Series(gain).ewm(span=period, min_periods=period, adjust=False).mean().values
    avg_loss = pd.Series(loss).ewm(span=period, min_periods=period, adjust=False).mean().values
    
    avg_gain = np.concatenate([[np.nan], avg_gain])
    avg_loss = np.concatenate([[np.nan], avg_loss])
    
    with np.errstate(divide='ignore', invalid='ignore'):
        rs = avg_gain / (avg_loss + 1e-10)
        rsi = 100 - (100 / (1 + rs))
    
    rsi = np.clip(rsi, 0, 100)
    return rsi

def calculate_rsi_streak(close, period=2):
    """Connors RSI Streak component - measures consecutive up/down days."""
    n = len(close)
    streak_rsi = np.full(n, np.nan)
    
    if n < period + 5:
        return streak_rsi
    
    streak = np.zeros(n)
    for i in range(1, n):
        if close[i] > close[i-1]:
            streak[i] = streak[i-1] + 1 if streak[i-1] >= 0 else 1
        elif close[i] < close[i-1]:
            streak[i] = streak[i-1] - 1 if streak[i-1] <= 0 else -1
        else:
            streak[i] = 0
    
    streak_abs = np.abs(streak)
    streak_score = np.where(streak >= 0, streak_abs, -streak_abs)
    streak_rsi = calculate_rsi(streak_score + 100, period)
    return streak_rsi

def calculate_percent_rank(close, period=100):
    """Connors RSI Percent Rank component."""
    n = len(close)
    pct_rank = np.full(n, np.nan)
    
    if n < period + 1:
        return pct_rank
    
    returns = np.diff(close) / (close[:-1] + 1e-10) * 100
    returns = np.concatenate([[0], returns])
    
    for i in range(period, n):
        window = returns[i-period:i]
        current = returns[i]
        pct_rank[i] = 100 * np.sum(window < current) / period
    
    return pct_rank

def calculate_crsi(close, rsi_period=3, streak_period=2, pct_rank_period=100):
    """Connors RSI = (RSI(3) + RSI_Streak(2) + PercentRank(100)) / 3"""
    rsi_short = calculate_rsi(close, rsi_period)
    streak_rsi = calculate_rsi_streak(close, streak_period)
    pct_rank = calculate_percent_rank(close, pct_rank_period)
    
    with np.errstate(invalid='ignore'):
        crsi = (rsi_short + streak_rsi + pct_rank) / 3
    
    crsi = np.clip(crsi, 0, 100)
    return crsi

def calculate_choppiness(high, low, close, period=14):
    """
    Choppiness Index - measures market choppiness vs trending.
    CHOP > 61.8 = range/choppy market (mean reversion)
    CHOP < 38.2 = trending market (trend following)
    """
    n = len(close)
    chop = np.full(n, np.nan)
    
    if n < period + 1:
        return chop
    
    tr = np.zeros(n)
    tr[0] = high[0] - low[0]
    for i in range(1, n):
        tr[i] = max(high[i] - low[i], np.abs(high[i] - close[i-1]), np.abs(low[i] - close[i-1]))
    
    for i in range(period, n):
        tr_sum = np.sum(tr[i-period+1:i+1])
        high_high = np.max(high[i-period+1:i+1])
        low_low = np.min(low[i-period+1:i+1])
        
        if high_high - low_low > 1e-10:
            chop[i] = 100 * np.log10(tr_sum / (high_high - low_low)) / np.log10(period)
        else:
            chop[i] = 50.0
    
    chop = np.clip(chop, 0, 100)
    return chop

def calculate_donchian(high, low, period=20):
    """Donchian Channel - highest high and lowest low over period."""
    n = len(close)
    upper = np.full(n, np.nan)
    lower = np.full(n, np.nan)
    
    for i in range(period - 1, n):
        upper[i] = np.max(high[i-period+1:i+1])
        lower[i] = np.min(low[i-period+1:i+1])
    
    return upper, lower

def calculate_atr(high, low, close, period=14):
    """Average True Range."""
    n = len(close)
    atr = np.full(n, np.nan)
    
    if n < period + 1:
        return atr
    
    tr = np.zeros(n)
    tr[0] = high[0] - low[0]
    for i in range(1, n):
        tr[i] = max(high[i] - low[i], np.abs(high[i] - close[i-1]), np.abs(low[i] - close[i-1]))
    
    atr = pd.Series(tr).ewm(span=period, min_periods=period, adjust=False).mean().values
    return atr

def generate_signals(prices):
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    n = len(close)
    
    # Load HTF data ONCE before loop (Rule 1 - CRITICAL)
    df_1w = get_htf_data(prices, '1w')
    
    # Calculate primary (1d) indicators
    crsi_1d = calculate_crsi(close, rsi_period=3, streak_period=2, pct_rank_period=100)
    atr_1d = calculate_atr(high, low, close, period=14)
    chop_1d = calculate_choppiness(high, low, close, period=14)
    donchian_upper, donchian_lower = calculate_donchian(high, low, period=20)
    
    # Calculate and align HTF HMA for trend bias
    hma_1w_raw = calculate_hma(df_1w['close'].values, 21)
    hma_1w_aligned = align_htf_to_ltf(prices, df_1w, hma_1w_raw)
    
    signals = np.zeros(n)
    BASE_SIZE = 0.30
    REDUCED_SIZE = 0.20
    
    # Position tracking for stoploss
    in_position = False
    position_side = 0
    entry_price = 0.0
    entry_atr = 0.0
    highest_since_entry = 0.0
    lowest_since_entry = float('inf')
    
    for i in range(150, n):
        # Skip if indicators not ready
        if np.isnan(crsi_1d[i]) or np.isnan(atr_1d[i]) or atr_1d[i] <= 1e-10:
            continue
        if np.isnan(chop_1d[i]) or np.isnan(donchian_upper[i]) or np.isnan(donchian_lower[i]):
            continue
        if np.isnan(hma_1w_aligned[i]):
            continue
        
        # === TREND BIAS (1w HTF HMA21) ===
        trend_1w_bullish = close[i] > hma_1w_aligned[i]
        trend_1w_bearish = close[i] < hma_1w_aligned[i]
        
        # === REGIME DETECTION (Choppiness Index) ===
        choppy_regime = chop_1d[i] > 61.8  # range market
        trending_regime = chop_1d[i] < 38.2  # trend market
        # neutral regime: 38.2 <= CHOP <= 61.8
        
        # === CRSI SIGNALS ===
        crsi_oversold = crsi_1d[i] < 15
        crsi_overbought = crsi_1d[i] > 85
        crsi_extreme_oversold = crsi_1d[i] < 10
        crsi_extreme_overbought = crsi_1d[i] > 90
        
        # === DONCHIAN BREAKOUT ===
        donchian_breakout_long = close[i] >= donchian_upper[i] * 0.998  # near or above upper
        donchian_breakout_short = close[i] <= donchian_lower[i] * 1.002  # near or below lower
        
        # === PRICE POSITION ===
        price_vs_donchian_mid = (donchian_upper[i] + donchian_lower[i]) / 2
        price_above_mid = close[i] > price_vs_donchian_mid
        price_below_mid = close[i] < price_vs_donchian_mid
        
        desired_signal = 0.0
        
        # === CHOPPY REGIME LOGIC (Mean Reversion) ===
        if choppy_regime:
            # Long: CRSI oversold + price above weekly HMA (trend filter)
            if crsi_oversold and trend_1w_bullish:
                desired_signal = BASE_SIZE
            
            # Short: CRSI overbought + price below weekly HMA (trend filter)
            if crsi_overbought and trend_1w_bearish:
                desired_signal = -BASE_SIZE
            
            # Extreme CRSI entries (higher conviction)
            if crsi_extreme_oversold and price_above_mid:
                desired_signal = BASE_SIZE
            
            if crsi_extreme_overbought and price_below_mid:
                desired_signal = -BASE_SIZE
        
        # === TRENDING REGIME LOGIC (Breakout) ===
        elif trending_regime:
            # Long breakout: Donchian upper + weekly bullish
            if donchian_breakout_long and trend_1w_bullish:
                desired_signal = BASE_SIZE
            
            # Short breakout: Donchian lower + weekly bearish
            if donchian_breakout_short and trend_1w_bearish:
                desired_signal = -BASE_SIZE
            
            # Pullback entries in trend
            if trend_1w_bullish and crsi_1d[i] < 40 and price_below_mid:
                desired_signal = REDUCED_SIZE
            
            if trend_1w_bearish and crsi_1d[i] > 60 and price_above_mid:
                desired_signal = -REDUCED_SIZE
        
        # === NEUTRAL REGIME (38.2 <= CHOP <= 61.8) ===
        else:
            # Conservative: only extreme CRSI + strong trend alignment
            if crsi_extreme_oversold and trend_1w_bullish:
                desired_signal = REDUCED_SIZE
            
            if crsi_extreme_overbought and trend_1w_bearish:
                desired_signal = -REDUCED_SIZE
            
            # Donchian breakout with trend confirmation
            if donchian_breakout_long and trend_1w_bullish:
                desired_signal = REDUCED_SIZE
            
            if donchian_breakout_short and trend_1w_bearish:
                desired_signal = -REDUCED_SIZE
        
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
        
        # === HOLD LOGIC — Maintain position if conditions intact ===
        if in_position and desired_signal == 0.0 and not stoploss_triggered:
            if position_side > 0:
                # Hold long if weekly trend intact and CRSI not overbought
                if trend_1w_bullish and crsi_1d[i] < 80:
                    desired_signal = BASE_SIZE
            elif position_side < 0:
                # Hold short if weekly trend intact and CRSI not oversold
                if trend_1w_bearish and crsi_1d[i] > 20:
                    desired_signal = -BASE_SIZE
        
        # === EXIT CONDITIONS ===
        if in_position and position_side > 0:
            # Exit long if weekly trend reverses or CRSI overbought
            if trend_1w_bearish and crsi_1d[i] > 70:
                desired_signal = 0.0
            # Exit if choppy regime and CRSI neutral (take profit)
            if choppy_regime and 40 < crsi_1d[i] < 60:
                desired_signal = 0.0
        
        if in_position and position_side < 0:
            # Exit short if weekly trend reverses or CRSI oversold
            if trend_1w_bullish and crsi_1d[i] < 30:
                desired_signal = 0.0
            # Exit if choppy regime and CRSI neutral (take profit)
            if choppy_regime and 40 < crsi_1d[i] < 60:
                desired_signal = 0.0
        
        # === DISCRETIZE SIGNAL VALUES ===
        if desired_signal > 0:
            desired_signal = BASE_SIZE if desired_signal >= BASE_SIZE else REDUCED_SIZE
        elif desired_signal < 0:
            desired_signal = -BASE_SIZE if desired_signal <= -BASE_SIZE else -REDUCED_SIZE
        
        # === UPDATE POSITION TRACKING ===
        if desired_signal != 0.0:
            if not in_position:
                in_position = True
                position_side = int(np.sign(desired_signal))
                entry_price = close[i]
                entry_atr = atr_1d[i]
                highest_since_entry = close[i] if position_side > 0 else 0.0
                lowest_since_entry = close[i] if position_side < 0 else float('inf')
            elif np.sign(desired_signal) != position_side:
                position_side = int(np.sign(desired_signal))
                entry_price = close[i]
                entry_atr = atr_1d[i]
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
        
        signals[i] = desired_signal
    
    return signals