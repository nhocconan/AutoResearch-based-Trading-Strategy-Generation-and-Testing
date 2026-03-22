#!/usr/bin/env python3
"""
Experiment #268: 30m Primary + 4h/1d HTF — Connors RSI + Choppiness Regime + Session Filter

Hypothesis: After #258 failed on 30m (Sharpe=-3.162), the issue was too many trades + weak filters.
This strategy uses:
1. 1d HMA(21) for PRIMARY trend direction (slow, reduces whipsaws)
2. 4h Choppiness(14) for regime detection (range vs trend)
3. 30m Connors RSI for entry timing (proven 75% win rate for mean reversion)
4. Session filter: only trade 8-20 UTC (highest volume, avoid overnight chop)
5. Volume filter: volume > 0.8x 20-bar average
6. Smaller position size (0.20 base, 0.30 strong) for lower TF
7. Tighter stoploss (2.0 * ATR) to reduce drawdown

Key differences from failed #258:
- Connors RSI instead of regular RSI (more sensitive to reversals)
- Session filter to avoid low-volume periods
- Volume confirmation on every entry
- Smaller position sizes (0.20-0.30 vs 0.35)
- Stricter regime alignment required

Position sizing: 0.20 base, 0.30 strong (discrete levels)
Target: 40-80 trades/year per symbol (appropriate for 30m with filters)
Stoploss: 2.0 * ATR trailing
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_30m_connors_chop_session_hma_4h1d_v1"
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
    """Calculate RSI using standard Wilder's method."""
    close_s = pd.Series(close)
    delta = close_s.diff()
    gain = delta.where(delta > 0, 0.0)
    loss = -delta.where(delta < 0, 0.0)
    
    avg_gain = gain.ewm(span=period, min_periods=period, adjust=False).mean()
    avg_loss = loss.ewm(span=period, min_periods=period, adjust=False).mean()
    
    rs = avg_gain / avg_loss.replace(0, np.nan)
    rsi = 100 - (100 / (1 + rs))
    rsi = rsi.fillna(50).values
    return rsi

def calculate_connors_rsi(close, rsi_period=3, streak_period=2, pr_period=100):
    """
    Calculate Connors RSI (CRSI).
    CRSI = (RSI(3) + RSI_Streak(2) + PercentRank(100)) / 3
    
    Components:
    1. RSI(3) - short-term momentum
    2. RSI of streak duration - measures how long trend has persisted
    3. PercentRank(100) - where current price ranks vs last 100 bars
    
    Entry signals:
    - Long: CRSI < 10 (extreme oversold)
    - Short: CRSI > 90 (extreme overbought)
    """
    n = len(close)
    close_s = pd.Series(close)
    
    # Component 1: RSI(3)
    rsi_short = calculate_rsi(close, rsi_period)
    
    # Component 2: RSI of Streak Duration
    # Calculate streak: consecutive up/down days
    delta = close_s.diff()
    streak = np.zeros(n)
    for i in range(1, n):
        if delta.iloc[i] > 0:
            streak[i] = streak[i-1] + 1 if streak[i-1] >= 0 else 1
        elif delta.iloc[i] < 0:
            streak[i] = streak[i-1] - 1 if streak[i-1] <= 0 else -1
        else:
            streak[i] = 0
    
    # Convert streak to absolute duration for RSI calculation
    streak_duration = np.abs(streak)
    streak_direction = np.sign(streak)
    
    # RSI of streak (treat streak duration as "price")
    streak_series = pd.Series(streak_duration)
    streak_delta = streak_series.diff()
    streak_gain = streak_delta.where(streak_delta > 0, 0.0)
    streak_loss = -streak_delta.where(streak_delta < 0, 0.0)
    
    avg_streak_gain = streak_gain.ewm(span=streak_period, min_periods=streak_period, adjust=False).mean()
    avg_streak_loss = streak_loss.ewm(span=streak_period, min_periods=streak_period, adjust=False).mean()
    
    streak_rs = avg_streak_gain / avg_streak_loss.replace(0, np.nan)
    rsi_streak = 100 - (100 / (1 + streak_rs))
    rsi_streak = rsi_streak.fillna(50).values
    
    # For mean reversion: invert streak RSI based on direction
    # Positive streak (uptrend) → high rsi_streak → want to short
    # Negative streak (downtrend) → low rsi_streak → want to long
    # So we use: 100 - rsi_streak when streak > 0, rsi_streak when streak < 0
    rsi_streak_adjusted = np.where(streak_direction > 0, 100 - rsi_streak, rsi_streak)
    
    # Component 3: PercentRank(100)
    percent_rank = np.zeros(n)
    for i in range(pr_period, n):
        window = close[i-pr_period:i]
        current = close[i]
        rank = np.sum(window < current) / pr_period * 100
        percent_rank[i] = rank
    
    # Combine components
    crsi = (rsi_short + rsi_streak_adjusted + percent_rank) / 3.0
    
    # Handle NaN
    crsi = np.nan_to_num(crsi, nan=50.0)
    crsi = np.clip(crsi, 0, 100)
    
    return crsi

def calculate_hma(close, period=21):
    """
    Calculate Hull Moving Average (HMA).
    HMA = WMA(2*WMA(n/2) - WMA(n), sqrt(n))
    """
    n = period
    half = n // 2
    sqrt_n = int(np.sqrt(n))
    
    if sqrt_n < 1:
        sqrt_n = 1
    
    close_s = pd.Series(close)
    
    def wma(series, window):
        if window < 1:
            window = 1
        weights = np.arange(1, window + 1)
        return series.rolling(window=window, min_periods=window).apply(
            lambda x: np.dot(x, weights) / weights.sum(), raw=True
        )
    
    wma_half = wma(close_s, half)
    wma_full = wma(close_s, n)
    
    raw_hma = 2 * wma_half - wma_full
    hma = wma(raw_hma, sqrt_n)
    
    return hma.values

def calculate_choppiness_index(high, low, close, period=14):
    """
    Calculate Choppiness Index (CHOP).
    CHOP > 61.8 = choppy/range market (mean revert)
    CHOP < 38.2 = trending market (trend follow)
    """
    n = period
    atr_vals = calculate_atr(high, low, close, period)
    
    atr_sum = pd.Series(atr_vals).rolling(window=n, min_periods=n).sum().values
    hh = pd.Series(high).rolling(window=n, min_periods=n).max().values
    ll = pd.Series(low).rolling(window=n, min_periods=n).min().values
    
    chop = np.zeros(len(close))
    for i in range(n, len(close)):
        range_hl = hh[i] - ll[i]
        if range_hl > 0 and atr_sum[i] > 0:
            chop[i] = 100 * np.log10(atr_sum[i] / range_hl) / np.log10(n)
        else:
            chop[i] = 50.0
    
    return chop

def calculate_volume_avg(volume, period=20):
    """Calculate rolling average volume."""
    vol_avg = pd.Series(volume).rolling(window=period, min_periods=period).mean().values
    return vol_avg

def get_utc_hour(open_time):
    """Extract UTC hour from open_time (milliseconds timestamp)."""
    # open_time is in milliseconds
    ts = pd.to_datetime(open_time, unit='ms', utc=True)
    return ts.dt.hour.values

def generate_signals(prices):
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    volume = prices["volume"].values
    open_time = prices["open_time"].values
    n = len(close)
    
    # Load HTF data ONCE before loop (Rule 1 - CRITICAL)
    df_4h = get_htf_data(prices, '4h')
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate 1d HTF indicators (primary trend regime)
    hma_1d_21 = calculate_hma(df_1d['close'].values, 21)
    hma_1d_50 = calculate_hma(df_1d['close'].values, 50)
    
    # Calculate 4h HTF indicators
    chop_4h_14 = calculate_choppiness_index(
        df_4h['high'].values, 
        df_4h['low'].values, 
        df_4h['close'].values, 
        14
    )
    hma_4h_21 = calculate_hma(df_4h['close'].values, 21)
    
    # Align HTF to LTF (Rule 2 - auto shift(1))
    hma_1d_21_aligned = align_htf_to_ltf(prices, df_1d, hma_1d_21)
    hma_1d_50_aligned = align_htf_to_ltf(prices, df_1d, hma_1d_50)
    chop_4h_14_aligned = align_htf_to_ltf(prices, df_4h, chop_4h_14)
    hma_4h_21_aligned = align_htf_to_ltf(prices, df_4h, hma_4h_21)
    
    # Calculate 30m indicators
    atr_14 = calculate_atr(high, low, close, 14)
    crsi_3_2_100 = calculate_connors_rsi(close, 3, 2, 100)
    rsi_14 = calculate_rsi(close, 14)
    vol_avg_20 = calculate_volume_avg(volume, 20)
    utc_hour = get_utc_hour(open_time)
    
    signals = np.zeros(n)
    
    # Position sizing (Rule 4 - discrete, max 0.40, smaller for 30m)
    BASE_SIZE = 0.20
    STRONG_SIZE = 0.30
    
    # Track position state
    in_position = False
    position_side = 0
    entry_price = 0.0
    highest_price = 0.0
    lowest_price = 0.0
    last_trade_bar = -20
    
    for i in range(150, n):
        # Skip if indicators not ready
        if np.isnan(atr_14[i]) or atr_14[i] == 0:
            continue
        
        if np.isnan(hma_1d_21_aligned[i]) or np.isnan(hma_1d_50_aligned[i]):
            continue
        
        if np.isnan(chop_4h_14_aligned[i]) or np.isnan(hma_4h_21_aligned[i]):
            continue
        
        if np.isnan(crsi_3_2_100[i]) or np.isnan(rsi_14[i]):
            continue
        
        if np.isnan(vol_avg_20[i]) or vol_avg_20[i] == 0:
            continue
        
        # === SESSION FILTER (8-20 UTC only) ===
        # Avoid overnight chop and low-volume periods
        in_session = 8 <= utc_hour[i] <= 20
        
        # === VOLUME FILTER ===
        volume_ok = volume[i] > 0.8 * vol_avg_20[i]
        
        # === 1D TREND REGIME (primary direction filter) ===
        price_above_1d_hma21 = close[i] > hma_1d_21_aligned[i]
        price_below_1d_hma21 = close[i] < hma_1d_21_aligned[i]
        hma_1d_bullish = hma_1d_21_aligned[i] > hma_1d_50_aligned[i]
        hma_1d_bearish = hma_1d_21_aligned[i] < hma_1d_50_aligned[i]
        
        # === 4H CHOPPINESS REGIME ===
        # CHOP > 55 = range market (mean revert entries)
        # CHOP < 45 = trend market (trend follow entries)
        is_choppy = chop_4h_14_aligned[i] > 55.0
        is_trending = chop_4h_14_aligned[i] < 45.0
        
        # === 4H TREND ALIGNMENT ===
        price_above_4h_hma = close[i] > hma_4h_21_aligned[i]
        price_below_4h_hma = close[i] < hma_4h_21_aligned[i]
        
        # === CONNORS RSI SIGNALS ===
        # Extreme oversold: CRSI < 15
        # Extreme overbought: CRSI > 85
        # Moderate: 20-80
        crsi_extreme_oversold = crsi_3_2_100[i] < 15.0
        crsi_extreme_overbought = crsi_3_2_100[i] > 85.0
        crsi_oversold = crsi_3_2_100[i] < 25.0
        crsi_overbought = crsi_3_2_100[i] > 75.0
        
        # === ENTRY LOGIC ===
        new_signal = 0.0
        bars_since_last_trade = i - last_trade_bar
        
        # Only trade during session with volume confirmation
        if not (in_session and volume_ok):
            signals[i] = 0.0 if not in_position else signals[i-1] if i > 0 else 0.0
            # Continue tracking position for stoploss
            if in_position and position_side != 0:
                if position_side > 0:
                    if close[i] > highest_price:
                        highest_price = close[i]
                if position_side < 0:
                    if lowest_price == 0.0 or close[i] < lowest_price:
                        lowest_price = close[i]
            continue
        
        # MEAN REVERSION MODE (choppy 4h regime)
        if is_choppy:
            # LONG: Choppy + CRSI extreme oversold + 1d not strongly bearish
            if crsi_extreme_oversold and not (price_below_1d_hma21 and hma_1d_bearish):
                new_signal = BASE_SIZE
            # LONG: Choppy + CRSI oversold + 1d bullish
            elif crsi_oversold and price_above_1d_hma21:
                new_signal = BASE_SIZE
            
            # SHORT: Choppy + CRSI extreme overbought + 1d not strongly bullish
            if crsi_extreme_overbought and not (price_above_1d_hma21 and hma_1d_bullish):
                if new_signal == 0.0:
                    new_signal = -BASE_SIZE
            # SHORT: Choppy + CRSI overbought + 1d bearish
            elif crsi_overbought and price_below_1d_hma21:
                if new_signal == 0.0:
                    new_signal = -BASE_SIZE
        
        # TREND FOLLOWING MODE (trending 4h regime)
        if is_trending:
            # LONG: Trending + 1d bullish + 4h bullish + CRSI not overbought
            if hma_1d_bullish and price_above_4h_hma and crsi_3_2_100[i] < 70:
                new_signal = STRONG_SIZE
            # LONG: Trending + CRSI oversold pullback in uptrend
            elif hma_1d_bullish and crsi_oversold and price_above_1d_hma21:
                if new_signal == 0.0:
                    new_signal = BASE_SIZE
            
            # SHORT: Trending + 1d bearish + 4h bearish + CRSI not oversold
            if hma_1d_bearish and price_below_4h_hma and crsi_3_2_100[i] > 30:
                if new_signal == 0.0 or abs(new_signal) < STRONG_SIZE:
                    new_signal = -STRONG_SIZE
            # SHORT: Trending + CRSI overbought pullback in downtrend
            elif hma_1d_bearish and crsi_overbought and price_below_1d_hma21:
                if new_signal == 0.0:
                    new_signal = -BASE_SIZE
        
        # === FREQUENCY SAFEGUARD ===
        # Force trade if no signal for 20 bars (~10 hours on 30m)
        # But only if conditions are reasonable
        if bars_since_last_trade > 20 and new_signal == 0.0 and not in_position:
            if is_choppy and crsi_extreme_oversold and volume_ok:
                new_signal = BASE_SIZE * 0.7
            elif is_choppy and crsi_extreme_overbought and volume_ok:
                new_signal = -BASE_SIZE * 0.7
            elif is_trending and hma_1d_bullish and crsi_oversold:
                new_signal = BASE_SIZE * 0.7
            elif is_trending and hma_1d_bearish and crsi_overbought:
                new_signal = -BASE_SIZE * 0.7
        
        # === STOPLOSS LOGIC (Rule 6) - 2.0 * ATR trailing ===
        stoploss_triggered = False
        
        if in_position and position_side != 0:
            if position_side > 0:
                if close[i] > highest_price:
                    highest_price = close[i]
                stoploss_price = highest_price - 2.0 * atr_14[i]
                if close[i] < stoploss_price:
                    stoploss_triggered = True
            
            if position_side < 0:
                if lowest_price == 0.0 or close[i] < lowest_price:
                    lowest_price = close[i]
                stoploss_price = lowest_price + 2.0 * atr_14[i]
                if close[i] > stoploss_price:
                    stoploss_triggered = True
        
        # === REGIME REVERSAL EXIT ===
        regime_reversal = False
        if in_position and position_side != 0:
            # Long position but 1d regime turns strongly bearish
            if position_side > 0 and price_below_1d_hma21 and hma_1d_bearish:
                regime_reversal = True
            # Short position but 1d regime turns strongly bullish
            if position_side < 0 and price_above_1d_hma21 and hma_1d_bullish:
                regime_reversal = True
        
        if stoploss_triggered or regime_reversal:
            new_signal = 0.0
        
        # === UPDATE POSITION TRACKING ===
        if new_signal != 0.0:
            if not in_position:
                in_position = True
                position_side = np.sign(new_signal)
                entry_price = close[i]
                highest_price = close[i] if position_side > 0 else 0.0
                lowest_price = close[i] if position_side < 0 else 0.0
                last_trade_bar = i
            elif np.sign(new_signal) != position_side:
                position_side = np.sign(new_signal)
                entry_price = close[i]
                highest_price = close[i] if position_side > 0 else 0.0
                lowest_price = close[i] if position_side < 0 else 0.0
                last_trade_bar = i
        else:
            if in_position:
                in_position = False
                position_side = 0
                entry_price = 0.0
                highest_price = 0.0
                lowest_price = 0.0
                last_trade_bar = i
        
        signals[i] = new_signal
    
    return signals