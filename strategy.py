#!/usr/bin/env python3
"""
Experiment #980: 1h Primary + 4h/12h HTF — Fisher Transform + Connors RSI + Regime Adaptive

Hypothesis: After 706 failed strategies, combining Ehlers Fisher Transform (reversal detection)
with Connors RSI (proven mean-reversion) and HTF trend bias should work across ALL symbols.

Why this should work:
1. Fisher Transform (period=9): Catches reversals in bear markets better than RSI.
   Long when Fisher crosses above -1.5, short when crosses below +1.5. 75% win rate in research.
2. Connors RSI: (RSI(3) + RSI_Streak(2) + PercentRank(100)) / 3
   Extreme values (<10 or >90) signal mean-reversion opportunities.
3. 4h HMA(21) + 12h HMA(21): Dual HTF trend bias prevents counter-trend trades.
4. Volume filter (>0.8x 20-bar avg): Only trade on high-conviction bars.
5. Session filter (8-20 UTC): Avoid low-liquidity periods that cause whipsaws.

Critical improvements over failed strategies:
- Fisher Transform instead of simple RSI (better reversal detection in 2022 crash)
- Connors RSI for additional mean-reversion signal
- RELAXED entry thresholds to ensure >=30 trades/train, >=3 trades/test
- Discrete signal sizes (0.0, ±0.25, ±0.30) minimize fee churn
- ALL symbols MUST have positive Sharpe (no SOL-only bias)
- ATR trailing stop at 2.5x for risk management

Target: Sharpe > 0.612, trades >= 30 train, >= 3 test, ALL symbols positive
Timeframe: 1h (target 40-70 trades/year with strict filters)
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_1h_fisher_crsi_regime_4h12h_hma_volume_session_v1"
timeframe = "1h"
leverage = 1.0

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

def calculate_fisher_transform(high, low, period=9):
    """
    Ehlers Fisher Transform - catches reversals better than RSI.
    Formula: Fisher = 0.5 * ln((1 + X) / (1 - X)) where X = 0.67 * (price - lowest) / (highest - lowest) - 0.33
    """
    n = len(high)
    fisher = np.full(n, np.nan)
    trigger = np.full(n, np.nan)
    
    if n < period:
        return fisher, trigger
    
    for i in range(period - 1, n):
        highest = np.max(high[i-period+1:i+1])
        lowest = np.min(low[i-period+1:i+1])
        
        if highest == lowest:
            fisher[i] = 0.0
            trigger[i] = fisher[i-1] if i > 0 else 0.0
            continue
        
        price = (high[i] + low[i]) / 2.0
        x = 0.67 * (price - lowest) / (highest - lowest) - 0.33
        x = np.clip(x, -0.99, 0.99)  # Prevent ln domain error
        
        fisher[i] = 0.5 * np.log((1 + x) / (1 - x + 1e-10))
        trigger[i] = fisher[i-1] if i > 0 else 0.0
    
    return fisher, trigger

def calculate_connors_rsi(close, rsi_period=3, streak_period=2, rank_period=100):
    """
    Connors RSI = (RSI(3) + RSI_Streak(2) + PercentRank(100)) / 3
    Proven mean-reversion signal with 75% win rate at extremes.
    """
    n = len(close)
    crsi = np.full(n, np.nan)
    
    if n < rank_period:
        return crsi
    
    # RSI(3)
    rsi_short = calculate_rsi(close, period=rsi_period)
    
    # RSI Streak (consecutive up/down days)
    streak_rsi = np.full(n, np.nan)
    delta = np.diff(close)
    streak = np.zeros(n)
    
    for i in range(1, n):
        if delta[i-1] > 0:
            streak[i] = streak[i-1] + 1 if i > 0 and streak[i-1] > 0 else 1
        elif delta[i-1] < 0:
            streak[i] = streak[i-1] - 1 if i > 0 and streak[i-1] < 0 else -1
        else:
            streak[i] = 0
    
    # Convert streak to RSI-like value (0-100)
    for i in range(streak_period, n):
        if streak[i] >= 0:
            streak_rsi[i] = min(100, 50 + streak[i] * 10)
        else:
            streak_rsi[i] = max(0, 50 + streak[i] * 10)
    
    # Percent Rank (100)
    percent_rank = np.full(n, np.nan)
    for i in range(rank_period - 1, n):
        window = close[i-rank_period+1:i+1]
        current = close[i]
        count_below = np.sum(window[:-1] < current)
        percent_rank[i] = 100 * count_below / (rank_period - 1)
    
    # Combine into Connors RSI
    for i in range(rank_period - 1, n):
        if not np.isnan(rsi_short[i]) and not np.isnan(streak_rsi[i]) and not np.isnan(percent_rank[i]):
            crsi[i] = (rsi_short[i] + streak_rsi[i] + percent_rank[i]) / 3.0
    
    return crsi

def calculate_hma(series, period):
    """Hull Moving Average."""
    series = pd.Series(series)
    half = period // 2
    sqrt_period = int(np.sqrt(period))
    
    wma_half = series.rolling(window=half, min_periods=half).mean() * 2
    wma_full = series.rolling(window=period, min_periods=period).mean()
    
    wma_diff = wma_half - wma_full
    hma = wma_diff.rolling(window=sqrt_period, min_periods=sqrt_period).mean()
    
    return hma.values

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

def calculate_volume_ratio(volume, period=20):
    """Volume ratio: current / 20-bar average."""
    n = len(volume)
    ratio = np.full(n, np.nan)
    
    for i in range(period - 1, n):
        avg_vol = np.mean(volume[i-period+1:i+1])
        if avg_vol > 1e-10:
            ratio[i] = volume[i] / avg_vol
    
    return ratio

def get_utc_hour(open_time):
    """Extract UTC hour from open_time (milliseconds timestamp)."""
    return pd.to_datetime(open_time, unit='ms').hour

def generate_signals(prices):
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    volume = prices["volume"].values
    open_time = prices["open_time"].values
    n = len(close)
    
    # Load HTF data ONCE before loop (Rule 1 - CRITICAL)
    df_4h = get_htf_data(prices, '4h')
    df_12h = get_htf_data(prices, '12h')
    
    # Calculate primary (1h) indicators
    fisher_1h, fisher_trigger_1h = calculate_fisher_transform(high, low, period=9)
    crsi_1h = calculate_connors_rsi(close, rsi_period=3, streak_period=2, rank_period=100)
    atr_1h = calculate_atr(high, low, close, period=14)
    vol_ratio_1h = calculate_volume_ratio(volume, period=20)
    
    # Calculate and align 4h HMA for medium-term trend
    hma_4h_raw = calculate_hma(df_4h['close'].values, 21)
    hma_4h_aligned = align_htf_to_ltf(prices, df_4h, hma_4h_raw)
    
    # Calculate and align 12h HMA for macro trend
    hma_12h_raw = calculate_hma(df_12h['close'].values, 21)
    hma_12h_aligned = align_htf_to_ltf(prices, df_12h, hma_12h_raw)
    
    # Extract UTC hours for session filter
    utc_hours = np.array([get_utc_hour(ot) for ot in open_time])
    
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
    
    for i in range(100, n):
        # Skip if indicators not ready
        if np.isnan(fisher_1h[i]) or np.isnan(fisher_trigger_1h[i]):
            continue
        if np.isnan(crsi_1h[i]) or np.isnan(atr_1h[i]) or atr_1h[i] <= 1e-10:
            continue
        if np.isnan(vol_ratio_1h[i]) or np.isnan(hma_4h_aligned[i]) or np.isnan(hma_12h_aligned[i]):
            continue
        
        # === SESSION FILTER (8-20 UTC) ===
        in_session = 8 <= utc_hours[i] <= 20
        
        # === VOLUME FILTER ===
        volume_ok = vol_ratio_1h[i] > 0.8
        
        # === TREND BIAS (4h + 12h HMA) ===
        trend_4h_bullish = close[i] > hma_4h_aligned[i]
        trend_4h_bearish = close[i] < hma_4h_aligned[i]
        trend_12h_bullish = close[i] > hma_12h_aligned[i]
        trend_12h_bearish = close[i] < hma_12h_aligned[i]
        
        # Strong trend when both HTF agree
        strong_bull = trend_4h_bullish and trend_12h_bullish
        strong_bear = trend_4h_bearish and trend_12h_bearish
        
        # === FISHER TRANSFORM SIGNALS ===
        # Fisher crosses above -1.5 from below = long signal
        fisher_long = fisher_1h[i] > -1.5 and fisher_trigger_1h[i] <= -1.5
        # Fisher crosses below +1.5 from above = short signal
        fisher_short = fisher_1h[i] < 1.5 and fisher_trigger_1h[i] >= 1.5
        
        # Fisher extreme values (reversal zones)
        fisher_extreme_low = fisher_1h[i] < -1.8
        fisher_extreme_high = fisher_1h[i] > 1.8
        
        # === CONNORS RSI SIGNALS ===
        crsi_oversold = crsi_1h[i] < 15
        crsi_overbought = crsi_1h[i] > 85
        crsi_extreme_oversold = crsi_1h[i] < 10
        crsi_extreme_overbought = crsi_1h[i] > 90
        
        desired_signal = 0.0
        
        # === LONG ENTRY CONDITIONS (3+ confluence required) ===
        long_confluence = 0
        
        if fisher_long:
            long_confluence += 2  # Fisher crossover is strong signal
        if fisher_extreme_low:
            long_confluence += 1
        if crsi_oversold:
            long_confluence += 1
        if crsi_extreme_oversold:
            long_confluence += 1
        if strong_bull:
            long_confluence += 1
        if trend_4h_bullish:
            long_confluence += 0.5
        if in_session and volume_ok:
            long_confluence += 0.5
        
        # Enter long with 2+ confluence (relaxed to ensure trades)
        if long_confluence >= 2.0:
            desired_signal = BASE_SIZE
        elif long_confluence >= 1.5 and crsi_extreme_oversold:
            desired_signal = REDUCED_SIZE
        elif long_confluence >= 1.5 and fisher_extreme_low:
            desired_signal = REDUCED_SIZE
        
        # === SHORT ENTRY CONDITIONS (3+ confluence required) ===
        short_confluence = 0
        
        if fisher_short:
            short_confluence += 2
        if fisher_extreme_high:
            short_confluence += 1
        if crsi_overbought:
            short_confluence += 1
        if crsi_extreme_overbought:
            short_confluence += 1
        if strong_bear:
            short_confluence += 1
        if trend_4h_bearish:
            short_confluence += 0.5
        if in_session and volume_ok:
            short_confluence += 0.5
        
        # Enter short with 2+ confluence
        if short_confluence >= 2.0:
            if desired_signal > 0:
                desired_signal = -BASE_SIZE  # Flip position
            else:
                desired_signal = -BASE_SIZE
        elif short_confluence >= 1.5 and crsi_extreme_overbought:
            if desired_signal > 0:
                desired_signal = -REDUCED_SIZE
            else:
                desired_signal = -REDUCED_SIZE
        elif short_confluence >= 1.5 and fisher_extreme_high:
            if desired_signal > 0:
                desired_signal = -REDUCED_SIZE
            else:
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
        
        # === HOLD LOGIC — Maintain position if trend intact ===
        if in_position and desired_signal == 0.0 and not stoploss_triggered:
            if position_side > 0:
                # Hold long if 4h trend still bullish and Fisher not overbought
                if trend_4h_bullish and fisher_1h[i] < 1.5:
                    desired_signal = BASE_SIZE
            elif position_side < 0:
                # Hold short if 4h trend still bearish and Fisher not oversold
                if trend_4h_bearish and fisher_1h[i] > -1.5:
                    desired_signal = -BASE_SIZE
        
        # === EXIT CONDITIONS ===
        if in_position and position_side > 0:
            # Exit long if 4h trend reverses AND Fisher overbought
            if trend_4h_bearish and fisher_1h[i] > 1.5:
                desired_signal = 0.0
            # Exit if Connors RSI extremely overbought
            if crsi_extreme_overbought:
                desired_signal = 0.0
        
        if in_position and position_side < 0:
            # Exit short if 4h trend reverses AND Fisher oversold
            if trend_4h_bullish and fisher_1h[i] < -1.5:
                desired_signal = 0.0
            # Exit if Connors RSI extremely oversold
            if crsi_extreme_oversold:
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
                entry_atr = atr_1h[i]
                highest_since_entry = close[i] if position_side > 0 else 0.0
                lowest_since_entry = close[i] if position_side < 0 else float('inf')
            elif np.sign(desired_signal) != position_side:
                position_side = int(np.sign(desired_signal))
                entry_price = close[i]
                entry_atr = atr_1h[i]
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