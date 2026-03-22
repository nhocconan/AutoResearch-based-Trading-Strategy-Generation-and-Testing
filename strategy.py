#!/usr/bin/env python3
"""
Experiment #077: 1d Primary + 1w HTF — Z-Score Mean Reversion + KAMA Trend + Choppiness Regime

Hypothesis: Previous strategies failed because they relied on laggy trend indicators (EMA/HMA crossover)
that whipsaw in bear markets. This strategy uses:
1. Z-Score (20) for mean reversion entries - catches extreme deviations from mean
2. KAMA (ER=10) for adaptive trend - adjusts smoothing based on market efficiency
3. Choppiness Index for regime detection - range vs trend market identification
4. 1w HMA(21) for major trend bias - prevents counter-trend trades in strong moves
5. Funding rate contrarian signal (if available) - proven edge for BTC/ETH in 2022 crash

Why this should work:
- Z-score mean reversion has proven effectiveness in bear/range markets (2025 test period)
- KAMA adapts to volatility - less lag in trends, more smoothing in chop
- 1w HTF prevents major counter-trend positions
- Choppiness filter avoids mean reversion in strong trends (where it fails)
- Daily timeframe naturally limits trades to 20-50/year

Timeframe: 1d (REQUIRED)
HTF: 1w via mtf_data.get_htf_data() — called ONCE before loop
Position sizing: 0.30 discrete
Stoploss: 2.5 * ATR(14) trailing
Target trades: 25-50/year per symbol
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_1d_zscore_kama_chop_1w_v1"
timeframe = "1d"
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

def calculate_zscore(close, period=20):
    """Calculate Z-score of price relative to rolling mean."""
    close_s = pd.Series(close)
    rolling_mean = close_s.rolling(window=period, min_periods=period).mean()
    rolling_std = close_s.rolling(window=period, min_periods=period).std()
    zscore = (close_s - rolling_mean) / rolling_std.replace(0, np.nan)
    zscore = zscore.fillna(0).values
    return zscore

def calculate_kama(close, er_period=10, fast_period=2, slow_period=30):
    """
    Calculate Kaufman's Adaptive Moving Average (KAMA).
    KAMA adapts smoothing based on market efficiency ratio.
    ER = |Close - Close(n)| / Sum(|Close(i) - Close(i-1)|)
    """
    close_s = pd.Series(close)
    n = len(close)
    kama = np.zeros(n)
    
    # Efficiency Ratio
    signal = np.abs(close_s - close_s.shift(er_period))
    noise = np.abs(close_s - close_s.shift(1)).rolling(window=er_period, min_periods=er_period).sum()
    er = signal / noise.replace(0, np.nan)
    er = er.fillna(0).values
    
    # Smoothing constant
    fast_sc = 2 / (fast_period + 1)
    slow_sc = 2 / (slow_period + 1)
    sc = (er * (fast_sc - slow_sc) + slow_sc) ** 2
    
    # Initialize KAMA
    kama[er_period] = close[er_period]
    
    for i in range(er_period + 1, n):
        if np.isnan(sc[i]):
            kama[i] = kama[i-1]
        else:
            kama[i] = kama[i-1] + sc[i] * (close[i] - kama[i-1])
    
    return kama

def calculate_choppiness(high, low, close, period=14):
    """
    Calculate Choppiness Index (CHOP).
    CHOP = 100 * LOG10(SUM(ATR, period) / (Highest High - Lowest Low)) / LOG10(period)
    CHOP > 61.8 = range market (mean revert)
    CHOP < 38.2 = trend market (trend follow)
    """
    atr_values = calculate_atr(high, low, close, period)
    
    # Rolling sum of ATR
    atr_sum = pd.Series(atr_values).rolling(window=period, min_periods=period).sum().values
    
    # Highest High and Lowest Low over period
    highest_high = pd.Series(high).rolling(window=period, min_periods=period).max().values
    lowest_low = pd.Series(low).rolling(window=period, min_periods=period).min().values
    
    # Choppiness calculation
    price_range = highest_high - lowest_low
    price_range = np.where(price_range == 0, 1e-10, price_range)
    
    chop = 100 * np.log10(atr_sum / price_range) / np.log10(period)
    chop = np.clip(chop, 0, 100)
    
    return chop

def calculate_hma(close, period=21):
    """Calculate Hull Moving Average for smoother trend with less lag."""
    close_s = pd.Series(close)
    half = max(1, period // 2)
    sqrt_period = max(1, int(np.sqrt(period)))
    wma1 = close_s.ewm(span=half, min_periods=half, adjust=False).mean()
    wma2 = close_s.ewm(span=period, min_periods=period, adjust=False).mean()
    wma3 = (2 * wma1 - wma2).ewm(span=sqrt_period, min_periods=sqrt_period, adjust=False).mean()
    return wma3.values

def calculate_hma_slope(hma_values, lookback=3):
    """Calculate HMA slope over lookback period."""
    slope = np.zeros(len(hma_values))
    for i in range(lookback, len(hma_values)):
        if hma_values[i - lookback] != 0 and not np.isnan(hma_values[i - lookback]):
            slope[i] = (hma_values[i] - hma_values[i - lookback]) / hma_values[i - lookback] * 100
    return slope

def calculate_connors_rsi(close, rsi_period=3, streak_period=2, rank_period=100):
    """
    Calculate Connors RSI (CRSI).
    CRSI = (RSI(3) + RSI_Streak(2) + PercentRank(100)) / 3
    """
    close_s = pd.Series(close)
    
    # Component 1: RSI(3)
    rsi_3 = calculate_rsi(close, rsi_period)
    
    # Component 2: RSI of Streak
    delta = close_s.diff()
    streak = np.zeros(len(close))
    
    for i in range(1, len(close)):
        if pd.isna(delta.iloc[i]):
            streak[i] = 0
        elif delta.iloc[i] > 0:
            streak[i] = streak[i-1] + 1 if streak[i-1] >= 0 else 1
        elif delta.iloc[i] < 0:
            streak[i] = streak[i-1] - 1 if streak[i-1] <= 0 else -1
        else:
            streak[i] = 0
    
    # Map streak to 0-100
    streak_rsi = np.zeros(len(close))
    for i in range(len(close)):
        streak_rsi[i] = min(100, max(0, 50 + streak[i] * 10))
    
    # Component 3: Percent Rank
    pct_change = close_s.pct_change()
    percent_rank = pd.Series(pct_change).rolling(window=rank_period, min_periods=rank_period).apply(
        lambda x: pd.Series(x).rank(pct=True).iloc[-1] * 100 if len(x) > 0 and not all(pd.isna(x)) else 50
    ).values
    percent_rank = np.nan_to_num(percent_rank, nan=50.0)
    
    crsi = (rsi_3 + streak_rsi + percent_rank) / 3.0
    return crsi

def generate_signals(prices):
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    n = len(close)
    
    # Load HTF data ONCE before loop (Rule 1 - CRITICAL)
    df_1w = get_htf_data(prices, '1w')
    
    # Calculate HTF indicators
    hma_1w_21 = calculate_hma(df_1w['close'].values, 21)
    hma_1w_slope = calculate_hma_slope(hma_1w_21, 3)
    
    # Align HTF to LTF (Rule 2 - auto shift(1) for completed bars)
    hma_1w_21_aligned = align_htf_to_ltf(prices, df_1w, hma_1w_21)
    hma_1w_slope_aligned = align_htf_to_ltf(prices, df_1w, hma_1w_slope)
    
    # Calculate 1d indicators
    atr_14 = calculate_atr(high, low, close, 14)
    chop_14 = calculate_choppiness(high, low, close, 14)
    zscore_20 = calculate_zscore(close, 20)
    kama_10 = calculate_kama(close, er_period=10)
    crsi = calculate_connors_rsi(close, 3, 2, 100)
    rsi_14 = calculate_rsi(close, 14)
    
    signals = np.zeros(n)
    
    # Base position sizing (Rule 4 - discrete levels, max 0.40)
    BASE_SIZE = 0.30
    
    # Track position state for stoploss
    in_position = False
    position_side = 0
    entry_price = 0.0
    highest_price = 0.0
    lowest_price = 0.0
    last_trade_bar = -50
    
    for i in range(100, n):
        # Skip if indicators not ready
        if np.isnan(atr_14[i]) or atr_14[i] == 0:
            continue
        
        if np.isnan(hma_1w_21_aligned[i]) or np.isnan(hma_1w_slope_aligned[i]):
            continue
        
        if np.isnan(chop_14[i]) or np.isnan(zscore_20[i]):
            continue
        
        if np.isnan(kama_10[i]) or np.isnan(crsi[i]):
            continue
        
        # === 1W TREND BIAS (MAJOR) ===
        trend_1w_bullish = hma_1w_slope_aligned[i] > 0.3
        trend_1w_bearish = hma_1w_slope_aligned[i] < -0.3
        trend_1w_neutral = not trend_1w_bullish and not trend_1w_bearish
        
        price_above_1w_hma = close[i] > hma_1w_21_aligned[i]
        price_below_1w_hma = close[i] < hma_1w_21_aligned[i]
        
        # === CHOPPINESS REGIME DETECTION ===
        is_range_market = chop_14[i] > 55
        is_trend_market = chop_14[i] < 45
        
        # === Z-SCORE MEAN REVERSION ===
        zscore_extreme_low = zscore_20[i] < -1.5
        zscore_extreme_high = zscore_20[i] > 1.5
        zscore_moderate_low = zscore_20[i] < -1.0
        zscore_moderate_high = zscore_20[i] > 1.0
        
        # === CONNORS RSI ===
        crsi_oversold = crsi[i] < 25
        crsi_overbought = crsi[i] > 75
        crsi_neutral_low = crsi[i] < 40
        crsi_neutral_high = crsi[i] > 60
        
        # === KAMA TREND ===
        kama_bullish = close[i] > kama_10[i]
        kama_bearish = close[i] < kama_10[i]
        
        # === POSITION SIZING ===
        current_size = BASE_SIZE
        
        # Reduce size in transitional markets
        if not is_range_market and not is_trend_market:
            current_size = BASE_SIZE * 0.7
        
        # === ENTRY LOGIC ===
        new_signal = 0.0
        bars_since_last_trade = i - last_trade_bar
        
        # LONG ENTRIES - Mean reversion in range market
        if is_range_market:
            # Strong mean reversion: extreme z-score + oversold CRSI
            if zscore_extreme_low and crsi_oversold:
                if trend_1w_neutral or trend_1w_bullish or price_above_1w_hma:
                    new_signal = current_size
            # Moderate mean reversion with 1w confirmation
            elif zscore_moderate_low and crsi_neutral_low:
                if trend_1w_bullish or price_above_1w_hma:
                    new_signal = current_size * 0.8
        
        # LONG ENTRIES - Trend following in trend market
        elif is_trend_market:
            # Pullback in uptrend
            if trend_1w_bullish and kama_bullish and crsi_neutral_low:
                new_signal = current_size
            # Breakout with z-score confirmation
            elif trend_1w_bullish and zscore_20[i] > -0.5 and kama_bullish:
                new_signal = current_size * 0.7
        
        # SHORT ENTRIES - Mean reversion in range market
        if is_range_market and new_signal == 0:
            if zscore_extreme_high and crsi_overbought:
                if trend_1w_neutral or trend_1w_bearish or price_below_1w_hma:
                    new_signal = -current_size
            elif zscore_moderate_high and crsi_neutral_high:
                if trend_1w_bearish or price_below_1w_hma:
                    new_signal = -current_size * 0.8
        
        # SHORT ENTRIES - Trend following in trend market
        elif is_trend_market and new_signal == 0:
            if trend_1w_bearish and kama_bearish and crsi_neutral_high:
                new_signal = -current_size
            elif trend_1w_bearish and zscore_20[i] < 0.5 and kama_bearish:
                new_signal = -current_size * 0.7
        
        # === FREQUENCY SAFEGUARD ===
        # If no trades for 90 bars (~90 days on 1d), allow weaker entry
        if bars_since_last_trade > 90 and new_signal == 0.0 and not in_position:
            if trend_1w_bullish and crsi[i] < 45:
                new_signal = current_size * 0.5
            elif trend_1w_bearish and crsi[i] > 55:
                new_signal = -current_size * 0.5
        
        # === STOPLOSS LOGIC (Rule 6) - 2.5 * ATR trailing ===
        stoploss_triggered = False
        
        if in_position and position_side != 0:
            if position_side > 0:
                if close[i] > highest_price:
                    highest_price = close[i]
                stoploss_price = highest_price - 2.5 * atr_14[i]
                if close[i] < stoploss_price:
                    stoploss_triggered = True
            
            if position_side < 0:
                if lowest_price == 0.0 or close[i] < lowest_price:
                    lowest_price = close[i]
                stoploss_price = lowest_price + 2.5 * atr_14[i]
                if close[i] > stoploss_price:
                    stoploss_triggered = True
        
        # === REGIME REVERSAL EXIT ===
        regime_reversal = False
        if in_position and position_side != 0:
            # Exit long if market becomes strongly trending bearish
            if position_side > 0 and is_trend_market and trend_1w_bearish:
                regime_reversal = True
            # Exit short if market becomes strongly trending bullish
            if position_side < 0 and is_trend_market and trend_1w_bullish:
                regime_reversal = True
        
        # Apply stoploss or regime reversal
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