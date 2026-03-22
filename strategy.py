#!/usr/bin/env python3
"""
Experiment #061: 4h Primary + 1d/1w HTF — Connors RSI Mean Reversion with Choppiness Regime

Hypothesis: Previous 4h strategies failed because they used trend-following logic in 
range/bear markets (2022 crash, 2025 bear). This strategy uses:

1. CONNORS RSI (CRSI) for mean reversion entries - proven 75% win rate in research
   CRSI = (RSI(3) + RSI_Streak(2) + PercentRank(100)) / 3
   Long: CRSI < 15, Short: CRSI > 85

2. CHOPPINESS INDEX regime filter - switches between mean revert and trend follow
   CHOP(14) > 61.8 = range (use mean reversion)
   CHOP(14) < 38.2 = trend (use trend following)

3. 1d HMA(21) for major trend bias - only take longs if price > 1d HMA (and vice versa)

4. 1w HMA(21) for macro regime - reduce size in counter-macro trades

5. ATR(14) stoploss at 2.5x - wider for 4h timeframe volatility

6. Position size: 0.28 discrete (0.20 in counter-trend, 0.28 with trend)

Why this should work:
- Connors RSI catches oversold/overbought extremes in range markets
- Choppiness Index prevents mean reversion during strong trends (whipsaw protection)
- 1d/1w HTF prevents counter-trend trades in major moves
- 4h timeframe = 20-50 trades/year target (fee-efficient)
- Simpler entry conditions = more trades = better statistics

Timeframe: 4h (REQUIRED for this experiment)
HTF: 1d and 1w via mtf_data.get_htf_data() — called ONCE before loop
Position sizing: 0.20-0.28 discrete
Stoploss: 2.5 * ATR(14) trailing
Target trades: 20-50/year per symbol
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_4h_connors_chop_regime_1d1w_v1"
timeframe = "4h"
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
    CRSI = (RSI(close, 3) + RSI(streak, 2) + PercentRank(100)) / 3
    
    Streak RSI: RSI of consecutive up/down days
    PercentRank: percentile rank of today's return vs last 100 days
    """
    n = len(close)
    crsi = np.zeros(n)
    
    # RSI(3) on close
    rsi_close = calculate_rsi(close, rsi_period)
    
    # Streak calculation
    streak = np.zeros(n)
    for i in range(1, n):
        if close[i] > close[i-1]:
            streak[i] = streak[i-1] + 1 if streak[i-1] >= 0 else 1
        elif close[i] < close[i-1]:
            streak[i] = streak[i-1] - 1 if streak[i-1] <= 0 else -1
        else:
            streak[i] = streak[i-1]
    
    # RSI(2) on streak
    streak_s = pd.Series(streak)
    streak_delta = streak_s.diff()
    streak_gain = streak_delta.where(streak_delta > 0, 0.0)
    streak_loss = -streak_delta.where(streak_delta < 0, 0.0)
    
    avg_streak_gain = streak_gain.ewm(span=streak_period, min_periods=streak_period, adjust=False).mean()
    avg_streak_loss = streak_loss.ewm(span=streak_period, min_periods=streak_period, adjust=False).mean()
    
    streak_rs = avg_streak_gain / avg_streak_loss.replace(0, np.nan)
    rsi_streak = 100 - (100 / (1 + streak_rs))
    rsi_streak = rsi_streak.fillna(50).values
    
    # PercentRank(100) - percentile of today's return vs last 100 days
    returns = np.zeros(n)
    for i in range(1, n):
        returns[i] = (close[i] - close[i-1]) / close[i-1] * 100
    
    percent_rank = np.zeros(n)
    for i in range(pr_period, n):
        window = returns[i-pr_period:i]
        count_below = np.sum(window < returns[i])
        percent_rank[i] = count_below / pr_period * 100
    
    # Combine into CRSI
    for i in range(pr_period, n):
        crsi[i] = (rsi_close[i] + rsi_streak[i] + percent_rank[i]) / 3
    
    crsi[:pr_period] = 50  # Default for warmup
    return crsi

def calculate_choppiness(high, low, close, period=14):
    """
    Calculate Choppiness Index (CHOP).
    CHOP = 100 * LOG10(SUM(ATR, period) / (Highest High - Lowest Low)) / LOG10(period)
    
    CHOP > 61.8 = ranging market (mean reversion works)
    CHOP < 38.2 = trending market (trend following works)
    """
    n = len(close)
    chop = np.zeros(n)
    
    atr_values = calculate_atr(high, low, close, period)
    
    for i in range(period, n):
        atr_sum = np.sum(atr_values[i-period+1:i+1])
        highest_high = np.max(high[i-period+1:i+1])
        lowest_low = np.min(low[i-period+1:i+1])
        price_range = highest_high - lowest_low
        
        if price_range > 0 and atr_sum > 0:
            chop[i] = 100 * np.log10(atr_sum / price_range) / np.log10(period)
        else:
            chop[i] = 50
    
    chop[:period] = 50
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

def calculate_hma_slope(hma_values, lookback=5):
    """Calculate HMA slope over lookback period."""
    slope = np.zeros(len(hma_values))
    for i in range(lookback, len(hma_values)):
        if hma_values[i - lookback] != 0:
            slope[i] = (hma_values[i] - hma_values[i - lookback]) / hma_values[i - lookback] * 100
    return slope

def generate_signals(prices):
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    n = len(close)
    
    # Load HTF data ONCE before loop (Rule 1 - CRITICAL)
    df_1d = get_htf_data(prices, '1d')
    df_1w = get_htf_data(prices, '1w')
    
    # Calculate 1d HTF indicators
    hma_1d_21 = calculate_hma(df_1d['close'].values, 21)
    hma_1d_slope = calculate_hma_slope(hma_1d_21, 5)
    
    # Calculate 1w HTF indicators
    hma_1w_21 = calculate_hma(df_1w['close'].values, 21)
    
    # Align HTF to LTF (Rule 2 - auto shift(1) for completed bars)
    hma_1d_21_aligned = align_htf_to_ltf(prices, df_1d, hma_1d_21)
    hma_1d_slope_aligned = align_htf_to_ltf(prices, df_1d, hma_1d_slope)
    hma_1w_21_aligned = align_htf_to_ltf(prices, df_1w, hma_1w_21)
    
    # Calculate 4h indicators
    atr_14 = calculate_atr(high, low, close, 14)
    crsi = calculate_connors_rsi(close, rsi_period=3, streak_period=2, pr_period=100)
    chop_14 = calculate_choppiness(high, low, close, 14)
    
    # Additional trend indicators
    hma_4h_21 = calculate_hma(close, 21)
    hma_4h_48 = calculate_hma(close, 48)
    
    signals = np.zeros(n)
    
    # Base position sizing (Rule 4 - discrete levels, max 0.40)
    BASE_SIZE_TREND = 0.28
    BASE_SIZE_COUNTER = 0.20
    
    # Track position state for stoploss
    in_position = False
    position_side = 0
    entry_price = 0.0
    highest_price = 0.0
    lowest_price = 0.0
    last_trade_bar = -50
    
    for i in range(150, n):
        # Skip if indicators not ready
        if np.isnan(atr_14[i]) or atr_14[i] == 0:
            continue
        
        if np.isnan(hma_1d_21_aligned[i]) or np.isnan(hma_1w_21_aligned[i]):
            continue
        
        if np.isnan(crsi[i]) or np.isnan(chop_14[i]):
            continue
        
        if np.isnan(hma_4h_21[i]) or np.isnan(hma_4h_48[i]):
            continue
        
        # === 1D TREND BIAS (MAJOR) ===
        price_above_1d_hma = close[i] > hma_1d_21_aligned[i]
        price_below_1d_hma = close[i] < hma_1d_21_aligned[i]
        trend_1d_bullish = hma_1d_slope_aligned[i] > 0
        trend_1d_bearish = hma_1d_slope_aligned[i] < 0
        
        # === 1W MACRO REGIME ===
        price_above_1w_hma = close[i] > hma_1w_21_aligned[i]
        price_below_1w_hma = close[i] < hma_1w_21_aligned[i]
        
        # === CHOPPINESS REGIME ===
        # CHOP > 61.8 = range (mean reversion)
        # CHOP < 38.2 = trend (trend following)
        # 38.2 - 61.8 = neutral (use both)
        is_ranging = chop_14[i] > 61.8
        is_trending = chop_14[i] < 38.2
        is_neutral = not is_ranging and not is_trending
        
        # === 4H HMA TREND ===
        hma_bullish = hma_4h_21[i] > hma_4h_48[i]
        hma_bearish = hma_4h_21[i] < hma_4h_48[i]
        hma_bullish_cross = hma_4h_21[i] > hma_4h_48[i] and hma_4h_21[i-1] <= hma_4h_48[i-1]
        hma_bearish_cross = hma_4h_21[i] < hma_4h_48[i] and hma_4h_21[i-1] >= hma_4h_48[i-1]
        
        # === CONNORS RSI EXTREMES ===
        crsi_oversold = crsi[i] < 15
        crsi_overbought = crsi[i] > 85
        crsi_moderate_low = crsi[i] < 30
        crsi_moderate_high = crsi[i] > 70
        
        # === POSITION SIZING ===
        # Use smaller size for counter-trend trades
        long_size = BASE_SIZE_TREND if (price_above_1d_hma or trend_1d_bullish) else BASE_SIZE_COUNTER
        short_size = BASE_SIZE_TREND if (price_below_1d_hma or trend_1d_bearish) else BASE_SIZE_COUNTER
        
        # Reduce size in counter-macro (vs 1w)
        if price_below_1w_hma:
            long_size *= 0.7
        if price_above_1w_hma:
            short_size *= 0.7
        
        # === ENTRY LOGIC ===
        new_signal = 0.0
        bars_since_last_trade = i - last_trade_bar
        
        # LONG ENTRIES
        # Mean reversion in range: CRSI < 15 + price > 1d HMA
        if is_ranging:
            if crsi_oversold and price_above_1d_hma:
                new_signal = long_size
            elif crsi_moderate_low and price_above_1d_hma and hma_bullish:
                new_signal = long_size * 0.8
        
        # Trend following in trend: HMA cross + CRSI confirmation
        if is_trending:
            if hma_bullish_cross and crsi_moderate_low:
                new_signal = long_size
            elif hma_bullish and crsi_moderate_low and trend_1d_bullish:
                new_signal = long_size * 0.8
        
        # Neutral regime: use both signals
        if is_neutral:
            if crsi_moderate_low and price_above_1d_hma:
                new_signal = long_size * 0.8
            if hma_bullish_cross and crsi_moderate_low:
                new_signal = long_size
        
        # SHORT ENTRIES
        # Mean reversion in range: CRSI > 85 + price < 1d HMA
        if is_ranging:
            if crsi_overbought and price_below_1d_hma:
                new_signal = -short_size
            elif crsi_moderate_high and price_below_1d_hma and hma_bearish:
                new_signal = -short_size * 0.8
        
        # Trend following in trend: HMA cross + CRSI confirmation
        if is_trending:
            if hma_bearish_cross and crsi_moderate_high:
                new_signal = -short_size
            elif hma_bearish and crsi_moderate_high and trend_1d_bearish:
                new_signal = -short_size * 0.8
        
        # Neutral regime: use both signals
        if is_neutral:
            if crsi_moderate_high and price_below_1d_hma:
                new_signal = -short_size * 0.8
            if hma_bearish_cross and crsi_moderate_high:
                new_signal = -short_size
        
        # === FREQUENCY SAFEGUARD ===
        # If no trades for 100 bars (~17 days on 4h), allow weaker entry
        if bars_since_last_trade > 100 and new_signal == 0.0 and not in_position:
            if price_above_1d_hma and crsi_moderate_low:
                new_signal = long_size * 0.5
            elif price_below_1d_hma and crsi_moderate_high:
                new_signal = -short_size * 0.5
        
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
        
        # === TREND REVERSAL EXIT ===
        trend_reversal = False
        if in_position and position_side != 0:
            # Exit long if 1d trend strongly reverses
            if position_side > 0 and trend_1d_bearish and price_below_1d_hma:
                trend_reversal = True
            # Exit short if 1d trend strongly reverses
            if position_side < 0 and trend_1d_bullish and price_above_1d_hma:
                trend_reversal = True
        
        # === CRSI REVERSAL EXIT (take profit) ===
        crsi_exit = False
        if in_position and position_side != 0:
            # Exit long when CRSI becomes overbought
            if position_side > 0 and crsi[i] > 75:
                crsi_exit = True
            # Exit short when CRSI becomes oversold
            if position_side < 0 and crsi[i] < 25:
                crsi_exit = True
        
        # Apply stoploss or exits
        if stoploss_triggered or trend_reversal or crsi_exit:
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