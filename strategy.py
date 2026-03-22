#!/usr/bin/env python3
"""
Experiment #099: 4h Primary + 1d HTF — Fisher Transform + Choppiness Regime

Hypothesis: Previous 4h strategies failed due to overly strict entry conditions
(RSI extremes too rare) and regime filters that blocked most trades. Fisher Transform
is specifically designed for reversal detection in non-gaussian price distributions
and should trigger more frequently than RSI extremes. Combined with relaxed Choppiness
thresholds and simpler 1d trend bias, this should generate 30-60 trades/year.

Strategy Logic:
1. FISHER TRANSFORM (9): Long when Fisher crosses above -1.0, Short when crosses below +1.0
   - More sensitive than RSI, catches reversals earlier
2. CHOPPINESS INDEX (14): CHOP > 50 = favor mean reversion, CHOP < 50 = favor trend
   - Relaxed from 61.8/38.2 to allow more trades
3. 1d HMA(21) SLOPE: Simple bias (slope > 0 = prefer longs, slope < 0 = prefer shorts)
4. ATR(14) stoploss: 2.5x trailing stop
5. Position size: 0.30 discrete

Why this should work:
- Fisher Transform triggers 3-5x more often than RSI<20/>80 extremes
- Relaxed Choppiness thresholds (50 vs 61.8) allow more regime trades
- Simple 1d slope bias (not requiring extreme values)
- Fewer conflicting filters = more trades = better statistics
- 4h timeframe naturally limits to 30-60 trades/year

Timeframe: 4h (REQUIRED)
HTF: 1d via mtf_data.get_htf_data() — called ONCE before loop
Position sizing: 0.30 discrete
Stoploss: 2.5 * ATR(14) trailing
Target trades: 30-60/year per symbol
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_4h_fisher_chop_hma_1d_v1"
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

def calculate_fisher_transform(high, low, period=9):
    """
    Calculate Ehlers Fisher Transform.
    Converts price to a Gaussian normal distribution for clearer reversal signals.
    
    Fisher = 0.5 * ln((1 + X) / (1 - X))
    Where X = EMA((price - LL) / (HH - LL)) normalized to -0.99 to +0.99
    """
    high_s = pd.Series(high)
    low_s = pd.Series(low)
    close_s = pd.Series(low)  # Use low for more conservative signals
    
    # Calculate (close - LL) / (HH - LL) over period
    hh = high_s.rolling(window=period, min_periods=period).max()
    ll = low_s.rolling(window=period, min_periods=period).min()
    
    price_range = hh - ll
    price_range = price_range.replace(0, 1e-10)  # avoid div by zero
    
    x_raw = (close_s - ll) / price_range
    
    # EMA smoothing
    x_ema = x_raw.ewm(span=period, min_periods=period, adjust=False).mean()
    
    # Normalize to -0.99 to +0.99 to avoid ln(0) or ln(inf)
    x = np.clip(x_ema.values, -0.99, 0.99)
    
    # Fisher Transform
    fisher = 0.5 * np.log((1 + x) / (1 - x))
    fisher = np.nan_to_num(fisher, nan=0.0)
    
    return fisher

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
    price_range = np.where(price_range == 0, 1e-10, price_range)  # avoid div by zero
    
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
    """Calculate HMA slope over lookback period (percentage change)."""
    slope = np.zeros(len(hma_values))
    for i in range(lookback, len(hma_values)):
        if hma_values[i - lookback] != 0 and not np.isnan(hma_values[i - lookback]):
            slope[i] = (hma_values[i] - hma_values[i - lookback]) / np.abs(hma_values[i - lookback]) * 100
    return slope

def generate_signals(prices):
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    n = len(close)
    
    # Load HTF data ONCE before loop (Rule 1 - CRITICAL)
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate HTF indicators
    hma_1d_21 = calculate_hma(df_1d['close'].values, 21)
    hma_1d_slope = calculate_hma_slope(hma_1d_21, 3)
    
    # Align HTF to LTF (Rule 2 - auto shift(1) for completed bars)
    hma_1d_21_aligned = align_htf_to_ltf(prices, df_1d, hma_1d_21)
    hma_1d_slope_aligned = align_htf_to_ltf(prices, df_1d, hma_1d_slope)
    
    # Calculate 4h indicators
    atr_14 = calculate_atr(high, low, close, 14)
    chop_14 = calculate_choppiness(high, low, close, 14)
    fisher_9 = calculate_fisher_transform(high, low, 9)
    rsi_14 = calculate_rsi(close, 14)
    
    # HMA for trend confirmation
    hma_21 = calculate_hma(close, 21)
    hma_50 = calculate_hma(close, 50)
    
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
    
    # Track Fisher crossings for signal generation
    fisher_prev = 0.0
    
    for i in range(100, n):
        # Skip if indicators not ready
        if np.isnan(atr_14[i]) or atr_14[i] == 0:
            fisher_prev = fisher_9[i] if not np.isnan(fisher_9[i]) else fisher_prev
            continue
        
        if np.isnan(hma_1d_21_aligned[i]) or np.isnan(hma_1d_slope_aligned[i]):
            fisher_prev = fisher_9[i] if not np.isnan(fisher_9[i]) else fisher_prev
            continue
        
        if np.isnan(chop_14[i]) or np.isnan(fisher_9[i]):
            fisher_prev = fisher_9[i] if not np.isnan(fisher_9[i]) else fisher_prev
            continue
        
        if np.isnan(hma_21[i]) or np.isnan(hma_50[i]):
            fisher_prev = fisher_9[i] if not np.isnan(fisher_9[i]) else fisher_prev
            continue
        
        # === 1D TREND BIAS (MAJOR) ===
        # Simple slope bias - relaxed thresholds for more trades
        trend_1d_bullish = hma_1d_slope_aligned[i] > 0.0  # Any positive slope
        trend_1d_bearish = hma_1d_slope_aligned[i] < 0.0  # Any negative slope
        
        # Price vs 1d HMA for additional confirmation
        price_above_1d_hma = close[i] > hma_1d_21_aligned[i]
        price_below_1d_hma = close[i] < hma_1d_21_aligned[i]
        
        # === CHOPPINESS REGIME DETECTION (Relaxed thresholds) ===
        # CHOP > 50 = range market (mean revert strategy)
        # CHOP < 50 = trend market (trend follow strategy)
        is_range_market = chop_14[i] > 50
        is_trend_market = chop_14[i] < 50
        
        # === FISHER TRANSFORM SIGNALS ===
        # Long: Fisher crosses above -1.0 (oversold reversal)
        # Short: Fisher crosses below +1.0 (overbought reversal)
        fisher_cross_up = (fisher_9[i] > -1.0) and (fisher_prev <= -1.0)
        fisher_cross_down = (fisher_9[i] < 1.0) and (fisher_prev >= 1.0)
        
        # Also allow extreme Fisher values for stronger signals
        fisher_oversold = fisher_9[i] < -1.5
        fisher_overbought = fisher_9[i] > 1.5
        
        # === RSI CONFIRMATION (Loose thresholds) ===
        rsi_oversold = rsi_14[i] < 45  # Relaxed from 30
        rsi_overbought = rsi_14[i] > 55  # Relaxed from 70
        
        # === 4H TREND CONFIRMATION ===
        hma_bullish = hma_21[i] > hma_50[i]
        hma_bearish = hma_21[i] < hma_50[i]
        
        # === POSITION SIZING ===
        current_size = BASE_SIZE
        
        # Reduce size in weak trend conditions
        if not trend_1d_bullish and not trend_1d_bearish:
            current_size = BASE_SIZE * 0.7
        
        # === ENTRY LOGIC ===
        new_signal = 0.0
        bars_since_last_trade = i - last_trade_bar
        
        # LONG ENTRIES
        long_condition = False
        
        if is_range_market:
            # Mean reversion: Fisher reversal from oversold
            if fisher_cross_up or fisher_oversold:
                if rsi_oversold or price_above_1d_hma or trend_1d_bullish:
                    long_condition = True
        elif is_trend_market:
            # Trend following: Fisher reversal with trend bias
            if trend_1d_bullish and (fisher_cross_up or fisher_oversold):
                if hma_bullish or price_above_1d_hma:
                    long_condition = True
            # Also allow if price above 1d HMA with Fisher signal
            elif price_above_1d_hma and (fisher_cross_up or fisher_oversold):
                long_condition = True
        else:
            # Neutral: Fisher extreme with any 1d bias
            if fisher_oversold and (trend_1d_bullish or price_above_1d_hma):
                long_condition = True
        
        if long_condition:
            new_signal = current_size
        
        # SHORT ENTRIES
        short_condition = False
        
        if is_range_market:
            # Mean reversion: Fisher reversal from overbought
            if fisher_cross_down or fisher_overbought:
                if rsi_overbought or price_below_1d_hma or trend_1d_bearish:
                    short_condition = True
        elif is_trend_market:
            # Trend following: Fisher reversal with trend bias
            if trend_1d_bearish and (fisher_cross_down or fisher_overbought):
                if hma_bearish or price_below_1d_hma:
                    short_condition = True
            # Also allow if price below 1d HMA with Fisher signal
            elif price_below_1d_hma and (fisher_cross_down or fisher_overbought):
                short_condition = True
        else:
            # Neutral: Fisher extreme with any 1d bias
            if fisher_overbought and (trend_1d_bearish or price_below_1d_hma):
                short_condition = True
        
        if short_condition:
            new_signal = -current_size
        
        # === FREQUENCY SAFEGUARD ===
        # If no trades for 80 bars (~20 days on 4h), allow weaker entry
        if bars_since_last_trade > 80 and new_signal == 0.0 and not in_position:
            if fisher_oversold and (trend_1d_bullish or price_above_1d_hma):
                new_signal = current_size * 0.5
            elif fisher_overbought and (trend_1d_bearish or price_below_1d_hma):
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
        
        # === FISHER REVERSAL EXIT ===
        # Exit if Fisher reverses against position
        fisher_reversal = False
        if in_position and position_side != 0:
            # Exit long if Fisher crosses below 0
            if position_side > 0 and fisher_9[i] < 0 and fisher_prev >= 0:
                fisher_reversal = True
            # Exit short if Fisher crosses above 0
            if position_side < 0 and fisher_9[i] > 0 and fisher_prev <= 0:
                fisher_reversal = True
        
        # Apply stoploss or Fisher reversal
        if stoploss_triggered or fisher_reversal:
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
        
        # Update Fisher previous value
        fisher_prev = fisher_9[i]
    
    return signals