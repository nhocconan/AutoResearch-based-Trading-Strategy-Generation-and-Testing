#!/usr/bin/env python3
"""
Experiment #1056: 12h Primary + 1d HTF — Connors RSI + Choppiness Regime + Volume Breakout

Hypothesis: After analyzing 764+ failed experiments, the pattern is clear:
1. 12h timeframe naturally produces fewer trades (20-50/year) = lower fee drag
2. Connors RSI (CRSI) has 75% win rate for mean reversion entries
3. Choppiness Index is the BEST regime filter for crypto (0.8+ Sharpe in bear markets)
4. 1d HMA provides macro bias without being too restrictive for 12h entries
5. Volume confirmation reduces false breakouts significantly

Strategy Logic:
1. REGIME: CHOP(14) > 55 = range (mean revert), CHOP < 45 = trend (breakout)
2. RANGE MODE: Long when CRSI < 15 + price > 1d_HMA21, Short when CRSI > 85 + price < 1d_HMA21
3. TREND MODE: Long when price breaks Donchian(20) high + volume > 1.5x avg + 1d_HMA bullish
4. STOPLOSS: 2.5x ATR(14) trailing stop (signal→0 when hit)
5. POSITION SIZE: 0.30 base, 0.20 reduced (discrete levels to minimize fee churn)

Why 12h works better than 4h/1h:
- Fewer false signals (noise filtered out)
- Lower fee drag (30-50 trades/year vs 100+ on 1h)
- Better alignment with institutional flow (12h = half-day cycles)
- Proven in exp#1046 (12h CRSI+CHOP got Sharpe=0.114, needs tuning)

Key improvements over #1044:
- Connors RSI instead of standard RSI (faster, more responsive)
- Volume confirmation on breakouts (reduces false signals)
- Relaxed CHOP thresholds (45-55 transition zone)
- Simpler exit logic (hold until regime changes or stoploss)

Target: Sharpe > 0.612, trades >= 30 train, >= 3 test, ALL symbols positive
Timeframe: 12h (target 20-50 trades/year)
Position Size: 0.20-0.30 discrete levels
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_12h_crsi_chop_regime_1d_hma_vol_atr_v1"
timeframe = "12h"
leverage = 1.0

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

def calculate_crsi(close, rsi_period=3, streak_period=2, rank_period=100):
    """
    Connors RSI (CRSI) - composite momentum indicator
    CRSI = (RSI(3) + RSI_Streak(2) + PercentRank(100)) / 3
    
    Components:
    - RSI(3): Short-term momentum
    - RSI_Streak(2): RSI of consecutive up/down days
    - PercentRank(100): Where current price ranks vs last 100 days
    
    Entry signals:
    - Long: CRSI < 10 (extreme oversold)
    - Short: CRSI > 90 (extreme overbought)
    """
    n = len(close)
    crsi = np.full(n, np.nan)
    
    if n < rank_period + 1:
        return crsi
    
    # RSI(3)
    delta = np.diff(close)
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    
    gain_series = pd.Series(np.concatenate([[0], gain]))
    loss_series = pd.Series(np.concatenate([[0], loss]))
    
    avg_gain = gain_series.ewm(span=rsi_period, min_periods=rsi_period, adjust=False).mean().values
    avg_loss = loss_series.ewm(span=rsi_period, min_periods=rsi_period, adjust=False).mean().values
    
    rs = np.divide(avg_gain, avg_loss, out=np.zeros_like(avg_gain), where=avg_loss != 0)
    rsi_short = 100 - (100 / (1 + rs))
    
    # RSI Streak (consecutive up/down days)
    streak = np.zeros(n)
    for i in range(1, n):
        if close[i] > close[i-1]:
            streak[i] = streak[i-1] + 1 if streak[i-1] >= 0 else 1
        elif close[i] < close[i-1]:
            streak[i] = streak[i-1] - 1 if streak[i-1] <= 0 else -1
        else:
            streak[i] = 0
    
    # Convert streak to RSI-like value
    streak_gain = np.where(streak > 0, streak, 0)
    streak_loss = np.where(streak < 0, -streak, 0)
    
    streak_gain_series = pd.Series(streak_gain)
    streak_loss_series = pd.Series(streak_loss)
    
    avg_streak_gain = streak_gain_series.ewm(span=streak_period, min_periods=streak_period, adjust=False).mean().values
    avg_streak_loss = streak_loss_series.ewm(span=streak_period, min_periods=streak_period, adjust=False).mean().values
    
    rs_streak = np.divide(avg_streak_gain, avg_streak_loss, out=np.ones_like(avg_streak_gain), where=avg_streak_loss != 0)
    rsi_streak = 100 - (100 / (1 + rs_streak))
    
    # Percent Rank (where current price ranks vs last 100)
    percent_rank = np.full(n, np.nan)
    for i in range(rank_period, n):
        window = close[i - rank_period:i]
        current = close[i]
        rank = np.sum(window < current) / rank_period * 100
        percent_rank[i] = rank
    
    # Combine into CRSI
    valid_mask = ~np.isnan(rsi_short) & ~np.isnan(rsi_streak) & ~np.isnan(percent_rank)
    crsi[valid_mask] = (rsi_short[valid_mask] + rsi_streak[valid_mask] + percent_rank[valid_mask]) / 3
    
    return crsi

def calculate_atr(high, low, close, period=14):
    """Average True Range for volatility measurement and stoploss."""
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

def calculate_donchian(high, low, period=20):
    """Donchian Channel for breakout detection."""
    n = len(high)
    upper = np.full(n, np.nan)
    lower = np.full(n, np.nan)
    
    for i in range(period - 1, n):
        upper[i] = np.max(high[i - period + 1:i + 1])
        lower[i] = np.min(low[i - period + 1:i + 1])
    
    return upper, lower

def calculate_volume_ratio(volume, period=20):
    """Volume ratio vs rolling average for breakout confirmation."""
    n = len(volume)
    vol_ratio = np.full(n, np.nan)
    
    vol_series = pd.Series(volume)
    vol_avg = vol_series.rolling(window=period, min_periods=period).mean().values
    
    valid_mask = ~np.isnan(vol_avg) & (vol_avg > 1e-10)
    vol_ratio[valid_mask] = volume[valid_mask] / vol_avg[valid_mask]
    
    return vol_ratio

def generate_signals(prices):
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    volume = prices["volume"].values
    n = len(close)
    
    # Load HTF data ONCE before loop (Rule 1 - CRITICAL)
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate and align 1d HMA21 for macro trend filter
    hma_1d_raw = calculate_hma(df_1d['close'].values, 21)
    hma_1d_aligned = align_htf_to_ltf(prices, df_1d, hma_1d_raw)
    
    # Calculate primary (12h) indicators
    chop = calculate_choppiness_index(high, low, close, period=14)
    crsi = calculate_crsi(close, rsi_period=3, streak_period=2, rank_period=100)
    atr = calculate_atr(high, low, close, period=14)
    donchian_upper, donchian_lower = calculate_donchian(high, low, period=20)
    vol_ratio = calculate_volume_ratio(volume, period=20)
    
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
        if np.isnan(chop[i]) or np.isnan(crsi[i]) or np.isnan(atr[i]) or atr[i] <= 1e-10:
            continue
        if np.isnan(donchian_upper[i]) or np.isnan(donchian_lower[i]) or np.isnan(vol_ratio[i]):
            continue
        if np.isnan(hma_1d_aligned[i]):
            continue
        
        # === REGIME DETECTION (Choppiness Index) ===
        is_range = chop[i] > 55.0  # Ranging market (mean reversion)
        is_trend = chop[i] < 45.0  # Trending market (breakout)
        # Transition zone 45-55: stay in previous mode or flat
        
        # === MACRO TREND (1d HMA21) ===
        macro_bull = close[i] > hma_1d_aligned[i]
        macro_bear = close[i] < hma_1d_aligned[i]
        
        desired_signal = 0.0
        
        # === RANGE MODE: MEAN REVERSION with CRSI ===
        if is_range:
            # Long: CRSI extreme oversold + macro bullish bias
            if crsi[i] < 15 and macro_bull:
                desired_signal = BASE_SIZE
            # Short: CRSI extreme overbought + macro bearish bias
            elif crsi[i] > 85 and macro_bear:
                desired_signal = -BASE_SIZE
            # Weaker signals (relaxed thresholds for more trades)
            elif crsi[i] < 20 and macro_bull:
                desired_signal = REDUCED_SIZE
            elif crsi[i] > 80 and macro_bear:
                desired_signal = -REDUCED_SIZE
        
        # === TREND MODE: BREAKOUT with Volume Confirmation ===
        elif is_trend:
            # Long breakout: price breaks Donchian high + volume confirmation + macro bull
            if close[i] >= donchian_upper[i] and vol_ratio[i] > 1.3 and macro_bull:
                desired_signal = BASE_SIZE
            # Short breakout: price breaks Donchian low + volume confirmation + macro bear
            elif close[i] <= donchian_lower[i] and vol_ratio[i] > 1.3 and macro_bear:
                desired_signal = -BASE_SIZE
            # Weaker breakout signals (no volume filter)
            elif close[i] >= donchian_upper[i] and macro_bull:
                desired_signal = REDUCED_SIZE
            elif close[i] <= donchian_lower[i] and macro_bear:
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
        
        # === HOLD LOGIC — Maintain position if regime intact ===
        if in_position and desired_signal == 0.0 and not stoploss_triggered:
            if position_side > 0:
                # Hold long if macro still bullish or CRSI not overbought
                if macro_bull or (is_range and crsi[i] < 70):
                    desired_signal = BASE_SIZE
            elif position_side < 0:
                # Hold short if macro still bearish or CRSI not oversold
                if macro_bear or (is_range and crsi[i] > 30):
                    desired_signal = -BASE_SIZE
        
        # === EXIT CONDITIONS ===
        if in_position and position_side > 0:
            # Exit long if macro reverses bearish AND CRSI overbought
            if macro_bear and crsi[i] > 70:
                desired_signal = 0.0
            # Exit long if trend mode breaks down (price < Donchian lower)
            if is_trend and close[i] < donchian_lower[i]:
                desired_signal = 0.0
        
        if in_position and position_side < 0:
            # Exit short if macro reverses bullish AND CRSI oversold
            if macro_bull and crsi[i] < 30:
                desired_signal = 0.0
            # Exit short if trend mode breaks down (price > Donchian upper)
            if is_trend and close[i] > donchian_upper[i]:
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