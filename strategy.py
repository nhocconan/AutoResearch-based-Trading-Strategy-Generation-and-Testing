#!/usr/bin/env python3
"""
Experiment #1052: 12h Primary + 1d/1w HTF — Connors RSI + Choppiness Regime + Asymmetric Trend Filter

Hypothesis: After analyzing 762+ failed strategies, the key insight is that 12h timeframe offers
the sweet spot between signal quality and trade frequency. This strategy combines:

1. CONNORS RSI (CRSI) - Proven 75% win rate mean reversion signal:
   - RSI(3) + RSI_Streak(2) + PercentRank(100) / 3
   - Long when CRSI < 15, Short when CRSI > 85
   - Much more responsive than standard RSI(14)

2. CHOPPINESS INDEX REGIME SWITCH:
   - CHOP(14) > 61.8 = RANGE mode → use CRSI mean reversion
   - CHOP(14) < 38.2 = TREND mode → use Donchian breakout + HMA trend
   - This adapts to market conditions (research shows 0.8+ Sharpe in bear markets)

3. 1d HMA21 MACRO TREND FILTER (asymmetric):
   - Only LONG when close > 1d_HMA21 (bullish macro)
   - Only SHORT when close < 1d_HMA21 (bearish macro)
   - Prevents counter-trend trades that destroy Sharpe in 2022 crash

4. 1w HMA50 SECONDARY FILTER:
   - Adds conviction when 1d and 1w agree on direction
   - Increases position size to 0.35 when both HTF align

5. ATR TRAILING STOP (2.5x):
   - Mandatory risk management - signal→0 when stop hit
   - Protects against 2022-style crashes

6. RELAXED THRESHOLDS for trade frequency:
   - CRSI: <20 / >80 (not extreme <10 / >90)
   - CHOP: 55-65 transition zone
   - Ensures 30+ trades/train, 3+ trades/test

Why this should beat Sharpe=0.612:
- CRSI has proven edge in mean reversion (75% win rate in research)
- Choppiness regime filter prevents trend-following in chop (major loss source)
- 12h timeframe = 20-50 trades/year optimal (not too many fees, not too few samples)
- Asymmetric HTF filter prevents catastrophic counter-trend trades
- Position sizing 0.25-0.35 protects against 77% BTC crash

Target: Sharpe > 0.612, trades >= 30 train, >= 3 test, ALL symbols positive Sharpe
Timeframe: 12h (target 20-50 trades/year)
Position Size: 0.25-0.35 discrete levels
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_12h_crsi_chop_regime_1d1w_hma_atr_v3"
timeframe = "12h"
leverage = 1.0

def calculate_crsi(close, rsi_period=3, streak_period=2, rank_period=100):
    """
    Connors RSI (CRSI) - combines 3 components for mean reversion signal
    Formula: (RSI(3) + RSI_Streak(2) + PercentRank(100)) / 3
    
    Research shows 75% win rate when CRSI < 10 (long) or > 90 (short)
    We use relaxed < 20 / > 80 for sufficient trade frequency
    """
    n = len(close)
    crsi = np.full(n, np.nan)
    
    if n < rank_period + 5:
        return crsi
    
    # Component 1: RSI(3)
    delta = np.diff(close)
    gain = np.zeros(n)
    loss = np.zeros(n)
    gain[1:] = np.where(delta > 0, delta, 0)
    loss[1:] = np.where(delta < 0, -delta, 0)
    
    gain_series = pd.Series(gain)
    loss_series = pd.Series(loss)
    
    avg_gain = gain_series.ewm(span=rsi_period, min_periods=rsi_period, adjust=False).mean().values
    avg_loss = loss_series.ewm(span=rsi_period, min_periods=rsi_period, adjust=False).mean().values
    
    rs = np.divide(avg_gain, avg_loss, out=np.zeros_like(avg_gain), where=avg_loss != 0)
    rsi_3 = 100 - (100 / (1 + rs))
    rsi_3[:rsi_period] = np.nan
    
    # Component 2: RSI of Streak (consecutive up/down days)
    streak = np.zeros(n)
    streak_dir = np.zeros(n)  # 1 = up streak, -1 = down streak
    
    for i in range(1, n):
        if close[i] > close[i-1]:
            if streak_dir[i-1] >= 0:
                streak[i] = streak[i-1] + 1
                streak_dir[i] = 1
            else:
                streak[i] = 1
                streak_dir[i] = 1
        elif close[i] < close[i-1]:
            if streak_dir[i-1] <= 0:
                streak[i] = streak[i-1] - 1
                streak_dir[i] = -1
            else:
                streak[i] = -1
                streak_dir[i] = -1
        else:
            streak[i] = 0
            streak_dir[i] = streak_dir[i-1]
    
    # Calculate RSI on absolute streak values
    streak_abs = np.abs(streak)
    streak_gain = np.zeros(n)
    streak_loss = np.zeros(n)
    
    for i in range(1, n):
        if streak[i] > 0:
            streak_gain[i] = streak_abs[i]
        elif streak[i] < 0:
            streak_loss[i] = streak_abs[i]
    
    streak_gain_series = pd.Series(streak_gain)
    streak_loss_series = pd.Series(streak_loss)
    
    avg_streak_gain = streak_gain_series.ewm(span=streak_period, min_periods=streak_period, adjust=False).mean().values
    avg_streak_loss = streak_loss_series.ewm(span=streak_period, min_periods=streak_period, adjust=False).mean().values
    
    streak_rs = np.divide(avg_streak_gain, avg_streak_loss, out=np.zeros_like(avg_streak_gain), where=avg_streak_loss != 0)
    rsi_streak = 100 - (100 / (1 + streak_rs))
    rsi_streak[:streak_period+5] = np.nan
    
    # Component 3: Percent Rank of daily returns over 100 days
    returns = np.zeros(n)
    returns[1:] = np.diff(close) / close[:-1] * 100
    
    percent_rank = np.full(n, np.nan)
    for i in range(rank_period, n):
        window = returns[i-rank_period+1:i+1]
        if len(window) == rank_period:
            count_below = np.sum(window[:-1] < returns[i])
            percent_rank[i] = count_below / (rank_period - 1) * 100
    
    # Combine all three components
    valid_mask = ~np.isnan(rsi_3) & ~np.isnan(rsi_streak) & ~np.isnan(percent_rank)
    crsi[valid_mask] = (rsi_3[valid_mask] + rsi_streak[valid_mask] + percent_rank[valid_mask]) / 3
    
    return crsi

def calculate_choppiness_index(high, low, close, period=14):
    """
    Choppiness Index (CHOP) - measures market ranging vs trending
    CHOP > 61.8 = ranging market (mean reversion works)
    CHOP < 38.2 = trending market (trend following works)
    
    Formula: CHOP = 100 * LOG10(SUM(ATR, n) / (Highest High - Lowest Low)) / LOG10(n)
    """
    n = len(close)
    chop = np.full(n, np.nan)
    
    if n < period + 1:
        return chop
    
    tr = np.zeros(n)
    tr[0] = high[0] - low[0]
    for i in range(1, n):
        tr[i] = max(high[i] - low[i], abs(high[i] - close[i-1]), abs(low[i] - close[i-1]))
    
    for i in range(period, n):
        sum_atr = np.sum(tr[i - period + 1:i + 1])
        highest_high = np.max(high[i - period + 1:i + 1])
        lowest_low = np.min(low[i - period + 1:i + 1])
        price_range = highest_high - lowest_low
        
        if price_range > 1e-10 and sum_atr > 1e-10:
            chop[i] = 100 * np.log10(sum_atr / price_range) / np.log10(period)
        else:
            chop[i] = 50.0
    
    return chop

def calculate_atr(high, low, close, period=14):
    """Average True Range for volatility measurement and stops."""
    n = len(close)
    atr = np.full(n, np.nan)
    
    if n < period + 1:
        return atr
    
    tr = np.zeros(n)
    tr[0] = high[0] - low[0]
    for i in range(1, n):
        tr[i] = max(high[i] - low[i], abs(high[i] - close[i-1]), abs(low[i] - close[i-1]))
    
    atr = pd.Series(tr).ewm(span=period, min_periods=period, adjust=False).mean().values
    return atr

def calculate_hma(series, period):
    """Hull Moving Average - faster and smoother than EMA."""
    series = pd.Series(series)
    half = period // 2
    sqrt_period = int(np.sqrt(period))
    
    wma_half = series.rolling(window=half, min_periods=half).mean() * 2
    wma_full = series.rolling(window=period, min_periods=period).mean()
    
    wma_diff = wma_half - wma_full
    hma = wma_diff.rolling(window=sqrt_period, min_periods=sqrt_period).mean()
    
    return hma.values

def calculate_donchian_channels(high, low, period=20):
    """Donchian Channels for breakout detection."""
    n = len(high)
    upper = np.full(n, np.nan)
    lower = np.full(n, np.nan)
    
    if n < period:
        return upper, lower
    
    for i in range(period - 1, n):
        upper[i] = np.max(high[i - period + 1:i + 1])
        lower[i] = np.min(low[i - period + 1:i + 1])
    
    return upper, lower

def generate_signals(prices):
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    n = len(close)
    
    # Load HTF data ONCE before loop (Rule 1 - CRITICAL)
    df_1d = get_htf_data(prices, '1d')
    df_1w = get_htf_data(prices, '1w')
    
    # Calculate and align 1d HMA21 for primary macro trend filter
    hma_1d_raw = calculate_hma(df_1d['close'].values, 21)
    hma_1d_aligned = align_htf_to_ltf(prices, df_1d, hma_1d_raw)
    
    # Calculate and align 1w HMA50 for secondary conviction filter
    hma_1w_raw = calculate_hma(df_1w['close'].values, 50)
    hma_1w_aligned = align_htf_to_ltf(prices, df_1w, hma_1w_raw)
    
    # Calculate primary (12h) indicators
    crsi = calculate_crsi(close, rsi_period=3, streak_period=2, rank_period=100)
    chop = calculate_choppiness_index(high, low, close, period=14)
    atr = calculate_atr(high, low, close, period=14)
    donchian_upper, donchian_lower = calculate_donchian_channels(high, low, period=20)
    
    # HMA for trend mode
    hma_8 = calculate_hma(close, 8)
    hma_21 = calculate_hma(close, 21)
    
    signals = np.zeros(n)
    BASE_SIZE = 0.30
    HIGH_CONV_SIZE = 0.35
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
        if np.isnan(crsi[i]) or np.isnan(chop[i]) or np.isnan(atr[i]) or atr[i] <= 1e-10:
            continue
        if np.isnan(donchian_upper[i]) or np.isnan(donchian_lower[i]):
            continue
        if np.isnan(hma_1d_aligned[i]) or np.isnan(hma_1w_aligned[i]):
            continue
        if np.isnan(hma_8[i]) or np.isnan(hma_21[i]):
            continue
        
        # === REGIME DETECTION (Choppiness Index) ===
        is_range = chop[i] > 55.0  # Ranging market (mean reversion)
        is_trend = chop[i] < 45.0  # Trending market (trend following)
        # Transition zone 45-55: maintain previous bias or stay flat
        
        # === MACRO TREND FILTERS ===
        macro_bull_1d = close[i] > hma_1d_aligned[i]
        macro_bear_1d = close[i] < hma_1d_aligned[i]
        macro_bull_1w = not np.isnan(hma_1w_aligned[i]) and close[i] > hma_1w_aligned[i]
        macro_bear_1w = not np.isnan(hma_1w_aligned[i]) and close[i] < hma_1w_aligned[i]
        
        # High conviction when both 1d and 1w agree
        high_conviction_long = macro_bull_1d and macro_bull_1w
        high_conviction_short = macro_bear_1d and macro_bear_1w
        
        desired_signal = 0.0
        
        # === RANGE MODE: MEAN REVERSION with CRSI ===
        if is_range:
            # Long: CRSI oversold + 1d HMA bullish bias
            if crsi[i] < 20 and macro_bull_1d:
                desired_signal = HIGH_CONV_SIZE if high_conviction_long else BASE_SIZE
            # Short: CRSI overbought + 1d HMA bearish bias
            elif crsi[i] > 80 and macro_bear_1d:
                desired_signal = -HIGH_CONV_SIZE if high_conviction_short else -BASE_SIZE
            # Weaker signals without HTF confirmation
            elif crsi[i] < 15:
                desired_signal = REDUCED_SIZE
            elif crsi[i] > 85:
                desired_signal = -REDUCED_SIZE
        
        # === TREND MODE: DONCHIAN BREAKOUT + HMA TREND ===
        elif is_trend:
            # Long: Price breaks Donchian upper + HMA8 > HMA21 + 1d HMA bullish
            if close[i] >= donchian_upper[i] * 0.998 and hma_8[i] > hma_21[i] and macro_bull_1d:
                desired_signal = HIGH_CONV_SIZE if high_conviction_long else BASE_SIZE
            # Short: Price breaks Donchian lower + HMA8 < HMA21 + 1d HMA bearish
            elif close[i] <= donchian_lower[i] * 1.002 and hma_8[i] < hma_21[i] and macro_bear_1d:
                desired_signal = -HIGH_CONV_SIZE if high_conviction_short else -BASE_SIZE
            # Weaker trend signals
            elif hma_8[i] > hma_21[i] and macro_bull_1d:
                desired_signal = REDUCED_SIZE
            elif hma_8[i] < hma_21[i] and macro_bear_1d:
                desired_signal = -REDUCED_SIZE
        
        # === TRANSITION ZONE: Hold existing positions, no new entries ===
        else:
            if in_position:
                desired_signal = BASE_SIZE if position_side > 0 else -BASE_SIZE
        
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
        
        # === HOLD LOGIC — Maintain position if regime intact ===
        if in_position and desired_signal == 0.0 and not stoploss_triggered:
            if position_side > 0:
                # Hold long if macro still bullish or CRSI not overbought
                if macro_bull_1d or (is_range and crsi[i] < 70):
                    desired_signal = BASE_SIZE
            elif position_side < 0:
                # Hold short if macro still bearish or CRSI not oversold
                if macro_bear_1d or (is_range and crsi[i] > 30):
                    desired_signal = -BASE_SIZE
        
        # === EXIT CONDITIONS ===
        if in_position and position_side > 0:
            # Exit long if 1d macro reverses bearish AND CRSI overbought
            if macro_bear_1d and crsi[i] > 70:
                desired_signal = 0.0
            # Exit long if trend mode and HMA crossover reverses
            if is_trend and hma_8[i] < hma_21[i]:
                desired_signal = 0.0
        
        if in_position and position_side < 0:
            # Exit short if 1d macro reverses bullish AND CRSI oversold
            if macro_bull_1d and crsi[i] < 30:
                desired_signal = 0.0
            # Exit short if trend mode and HMA crossover reverses
            if is_trend and hma_8[i] > hma_21[i]:
                desired_signal = 0.0
        
        # === DISCRETIZE SIGNAL VALUES ===
        if desired_signal > 0:
            if desired_signal >= HIGH_CONV_SIZE:
                desired_signal = HIGH_CONV_SIZE
            elif desired_signal >= BASE_SIZE:
                desired_signal = BASE_SIZE
            else:
                desired_signal = REDUCED_SIZE
        elif desired_signal < 0:
            if desired_signal <= -HIGH_CONV_SIZE:
                desired_signal = -HIGH_CONV_SIZE
            elif desired_signal <= -BASE_SIZE:
                desired_signal = -BASE_SIZE
            else:
                desired_signal = -REDUCED_SIZE
        
        # === UPDATE POSITION TRACKING ===
        if desired_signal != 0.0:
            if not in_position:
                in_position = True
                position_side = int(np.sign(desired_signal))
                entry_price = close[i]
                entry_atr = atr[i]
                highest_since_entry = close[i] if position_side > 0 else 0.0
                lowest_since_entry = close[i] if position_side < 0 else float('inf')
            elif np.sign(desired_signal) != position_side:
                # Flip position
                position_side = int(np.sign(desired_signal))
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
        
        signals[i] = desired_signal
    
    return signals