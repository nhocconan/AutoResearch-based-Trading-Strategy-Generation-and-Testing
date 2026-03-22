#!/usr/bin/env python3
"""
Experiment #092: 12h Primary + 1d/1w HTF — KAMA Adaptive Trend + RSI Pullback

Hypothesis: Previous strategies failed because they used too many conflicting filters
or extreme RSI thresholds that rarely trigger. KAMA (Kaufman Adaptive Moving Average)
adapts to market noise better than HMA/EMA and should provide cleaner trend signals.
Combined with moderate RSI thresholds (35/65 not 20/80) and volume confirmation,
this should generate 30-50 trades/year with better win rate.

Strategy Logic:
1. KAMA(21) on 12h: Adaptive trend that smooths in chop, follows in trends
2. 1d KAMA(21) slope: Major trend bias via mtf_data (called ONCE before loop)
3. RSI(14): Entry timing with moderate thresholds (35/65) for more trades
4. Volume spike: volume > 1.5x 20-bar average confirms breakout validity
5. ATR(14) stoploss: 2.5x trailing stop on all positions
6. Position size: 0.30 discrete (balanced risk/opportunity)
7. Asymmetric bias: prefer longs when 1d KAMA slope > 0, shorts when < 0

Why this should work:
- KAMA adapts efficiency ratio to market conditions (better than static MA)
- Moderate RSI thresholds = more trades = better statistics
- Volume confirmation filters false breakouts
- 12h timeframe naturally limits to 30-50 trades/year
- Simpler logic = fewer failure points

Timeframe: 12h (REQUIRED)
HTF: 1d via mtf_data.get_htf_data() — called ONCE before loop
Position sizing: 0.30 discrete
Stoploss: 2.5 * ATR(14) trailing
Target trades: 30-50/year per symbol
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_12h_kama_rsi_vol_1d_v1"
timeframe = "12h"
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

def calculate_kama(close, period=10, fast_period=2, slow_period=30):
    """
    Calculate Kaufman Adaptive Moving Average (KAMA).
    KAMA adapts to market noise via Efficiency Ratio (ER).
    ER = |close - close_n| / sum(|close_i - close_i-1|)
    SC = (ER * (fast_sc - slow_sc) + slow_sc)^2
    KAMA = KAMA_prev + SC * (close - KAMA_prev)
    """
    close_s = pd.Series(close)
    n = len(close)
    
    # Efficiency Ratio
    change = np.abs(close - np.roll(close, period))
    change[0:period] = np.nan
    
    volatility = np.zeros(n)
    for i in range(period, n):
        volatility[i] = np.sum(np.abs(close[i-period+1:i+1] - np.roll(close[i-period+1:i+1], 1)))
    
    er = change / np.where(volatility == 0, 1e-10, volatility)
    er = np.nan_to_num(er, nan=0.0)
    
    # Smoothing Constant
    fast_sc = 2.0 / (fast_period + 1)
    slow_sc = 2.0 / (slow_period + 1)
    sc = np.power(er * (fast_sc - slow_sc) + slow_sc, 2)
    
    # KAMA calculation
    kama = np.zeros(n)
    kama[0] = close[0]
    
    # Initialize with SMA for first period values
    for i in range(1, min(period, n)):
        kama[i] = close[:i+1].mean()
    
    for i in range(period, n):
        kama[i] = kama[i-1] + sc[i] * (close[i] - kama[i-1])
    
    return kama

def calculate_kama_slope(kama_values, lookback=5):
    """Calculate KAMA slope over lookback period."""
    slope = np.zeros(len(kama_values))
    for i in range(lookback, len(kama_values)):
        if kama_values[i - lookback] != 0:
            slope[i] = (kama_values[i] - kama_values[i - lookback]) / kama_values[i - lookback] * 100
    return slope

def calculate_volume_ma(volume, period=20):
    """Calculate moving average of volume."""
    vol_s = pd.Series(volume)
    vol_ma = vol_s.rolling(window=period, min_periods=period).mean().values
    return vol_ma

def generate_signals(prices):
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    volume = prices["volume"].values
    n = len(close)
    
    # Load HTF data ONCE before loop (Rule 1 - CRITICAL)
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate HTF indicators
    kama_1d_21 = calculate_kama(df_1d['close'].values, period=10, fast_period=2, slow_period=30)
    kama_1d_slope = calculate_kama_slope(kama_1d_21, 5)
    
    # Align HTF to LTF (Rule 2 - auto shift(1) for completed bars)
    kama_1d_21_aligned = align_htf_to_ltf(prices, df_1d, kama_1d_21)
    kama_1d_slope_aligned = align_htf_to_ltf(prices, df_1d, kama_1d_slope)
    
    # Calculate 12h indicators
    atr_14 = calculate_atr(high, low, close, 14)
    rsi_14 = calculate_rsi(close, 14)
    kama_12h_21 = calculate_kama(close, period=10, fast_period=2, slow_period=30)
    kama_12h_50 = calculate_kama(close, period=10, fast_period=2, slow_period=30)
    kama_12h_50 = calculate_kama(close, period=20, fast_period=5, slow_period=50)
    vol_ma_20 = calculate_volume_ma(volume, 20)
    
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
        
        if np.isnan(kama_1d_21_aligned[i]) or np.isnan(kama_1d_slope_aligned[i]):
            continue
        
        if np.isnan(rsi_14[i]) or np.isnan(kama_12h_21[i]) or np.isnan(kama_12h_50[i]):
            continue
        
        if np.isnan(vol_ma_20[i]) or vol_ma_20[i] == 0:
            continue
        
        # === 1D TREND BIAS (MAJOR) ===
        # KAMA slope > 0.3 = bullish bias (prefer longs)
        # KAMA slope < -0.3 = bearish bias (prefer shorts)
        trend_1d_bullish = kama_1d_slope_aligned[i] > 0.3
        trend_1d_bearish = kama_1d_slope_aligned[i] < -0.3
        trend_1d_neutral = not trend_1d_bullish and not trend_1d_bearish
        
        # Price vs 1d KAMA for additional confirmation
        price_above_1d_kama = close[i] > kama_1d_21_aligned[i]
        price_below_1d_kama = close[i] < kama_1d_21_aligned[i]
        
        # === 12H TREND CONFIRMATION ===
        kama_bullish = kama_12h_21[i] > kama_12h_50[i]
        kama_bearish = kama_12h_21[i] < kama_12h_50[i]
        
        # === RSI SIGNALS (moderate thresholds for more trades) ===
        rsi_oversold = rsi_14[i] < 40
        rsi_overbought = rsi_14[i] > 60
        rsi_extreme_low = rsi_14[i] < 35
        rsi_extreme_high = rsi_14[i] > 65
        
        # === VOLUME CONFIRMATION ===
        vol_spike = volume[i] > 1.5 * vol_ma_20[i]
        vol_normal = volume[i] > 0.8 * vol_ma_20[i]
        
        # === POSITION SIZING ===
        current_size = BASE_SIZE
        
        # Reduce size against 1d trend
        if trend_1d_bearish and current_size > 0:
            current_size = BASE_SIZE * 0.7
        if trend_1d_bullish and current_size < 0:
            current_size = BASE_SIZE * 0.7
        
        # === ENTRY LOGIC ===
        new_signal = 0.0
        bars_since_last_trade = i - last_trade_bar
        
        # LONG ENTRIES (multiple conditions for flexibility)
        long_condition_1 = trend_1d_bullish and kama_bullish and rsi_oversold and vol_normal
        long_condition_2 = price_above_1d_kama and kama_bullish and rsi_extreme_low
        long_condition_3 = trend_1d_bullish and rsi_14[i] < 45 and vol_spike
        long_condition_4 = kama_bullish and rsi_extreme_low and vol_normal
        
        if long_condition_1 or long_condition_2 or long_condition_3 or long_condition_4:
            new_signal = current_size
        
        # SHORT ENTRIES (multiple conditions for flexibility)
        short_condition_1 = trend_1d_bearish and kama_bearish and rsi_overbought and vol_normal
        short_condition_2 = price_below_1d_kama and kama_bearish and rsi_extreme_high
        short_condition_3 = trend_1d_bearish and rsi_14[i] > 55 and vol_spike
        short_condition_4 = kama_bearish and rsi_extreme_high and vol_normal
        
        if short_condition_1 or short_condition_2 or short_condition_3 or short_condition_4:
            new_signal = -current_size
        
        # === FREQUENCY SAFEGUARD ===
        # If no trades for 80 bars (~40 days on 12h), allow weaker entry
        if bars_since_last_trade > 80 and new_signal == 0.0 and not in_position:
            if trend_1d_bullish and rsi_14[i] < 50:
                new_signal = current_size * 0.5
            elif trend_1d_bearish and rsi_14[i] > 50:
                new_signal = -current_size * 0.5
        
        # === REVERSAL LOGIC ===
        # If we have a position and opposite signal is strong, reverse
        if in_position and new_signal != 0.0:
            if position_side > 0 and new_signal < 0:
                # Long to short reversal
                pass  # Allow reversal
            elif position_side < 0 and new_signal > 0:
                # Short to long reversal
                pass  # Allow reversal
        
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
        
        # Apply stoploss
        if stoploss_triggered:
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
                # Reversal
                position_side = np.sign(new_signal)
                entry_price = close[i]
                highest_price = close[i] if position_side > 0 else 0.0
                lowest_price = close[i] if position_side < 0 else 0.0
                last_trade_bar = i
            # If same side, keep position (could add pyramiding here)
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