#!/usr/bin/env python3
"""
Experiment #025: 1h Volatility Spike + BB Mean Reversion + 4h/1d Trend Filter

Hypothesis: Volatility spikes followed by Bollinger Band mean reversion, filtered by 
HTF trend direction, will capture panic bottoms and rally tops. Key improvements over 
failed Connors RSI strategies:
1. Vol spike detection: ATR(7)/ATR(30) > 1.8 signals extreme moves ready to revert
2. BB mean reversion: Price outside BB(20, 2.0) + vol spike = high-probability reversal
3. HTF trend filter: 4h HMA slope + 1d HMA position for directional bias (less strict)
4. Session filter relaxed: 6-22 UTC (wider window for more trades)
5. Volume filter: > 0.7x avg (lower threshold to allow more entries)
6. RSI confirmation: RSI(14) < 35 for longs, > 65 for shorts (more lenient than CRSI)

Why this should work:
- Vol spikes indicate panic/euphoria extremes that typically revert
- BB bands capture statistical extremes (2 std dev)
- HTF filter prevents counter-trend but allows trades in both directions
- 1h TF with relaxed filters targets 40-80 trades/year (optimal range)
- Discrete sizing (0.20, 0.25, 0.30) minimizes churn costs

Timeframe: 1h (REQUIRED)
HTF: 4h and 1d via mtf_data helper (call ONCE before loop)
Position sizing: 0.20-0.30 discrete
Stoploss: 2.5 * ATR(14)
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_1h_vol_spike_bb_reversion_4h_1d_trend_v1"
timeframe = "1h"
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
    delta = np.diff(close)
    delta = np.insert(delta, 0, 0)
    
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    
    gain_avg = pd.Series(gain).ewm(span=period, min_periods=period, adjust=False).mean().values
    loss_avg = pd.Series(loss).ewm(span=period, min_periods=period, adjust=False).mean().values
    
    loss_avg = np.where(loss_avg == 0, 1e-10, loss_avg)
    rs = gain_avg / loss_avg
    rsi = 100 - (100 / (1 + rs))
    rsi = np.clip(rsi, 0, 100)
    return rsi

def calculate_bollinger_bands(close, period=20, std_dev=2.0):
    """Calculate Bollinger Bands."""
    close_s = pd.Series(close)
    sma = close_s.rolling(window=period, min_periods=period).mean().values
    std = close_s.rolling(window=period, min_periods=period).std().values
    upper = sma + std_dev * std
    lower = sma - std_dev * std
    return upper, lower, sma

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
    """Calculate HMA slope over lookback periods."""
    slope = np.zeros(len(hma_values))
    for i in range(lookback, len(hma_values)):
        slope[i] = (hma_values[i] - hma_values[i - lookback]) / hma_values[i - lookback] * 100
    return slope

def calculate_volume_avg(volume, period=20):
    """Calculate rolling average volume."""
    vol_s = pd.Series(volume)
    vol_avg = vol_s.rolling(window=period, min_periods=period).mean().values
    return vol_avg

def get_session_hour(open_time_ms):
    """Extract UTC hour from millisecond timestamp."""
    return (open_time_ms // (1000 * 60 * 60)) % 24

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
    
    # Calculate 4h indicators
    hma_4h_21 = calculate_hma(df_4h['close'].values, 21)
    hma_4h_21_aligned = align_htf_to_ltf(prices, df_4h, hma_4h_21)
    slope_4h = calculate_hma_slope(hma_4h_21_aligned, lookback=3)
    
    # Calculate 1d indicators
    hma_1d_21 = calculate_hma(df_1d['close'].values, 21)
    hma_1d_21_aligned = align_htf_to_ltf(prices, df_1d, hma_1d_21)
    
    # Calculate 1h indicators
    atr_14 = calculate_atr(high, low, close, 14)
    atr_7 = calculate_atr(high, low, close, 7)
    atr_30 = calculate_atr(high, low, close, 30)
    rsi_14 = calculate_rsi(close, 14)
    bb_upper, bb_lower, bb_mid = calculate_bollinger_bands(close, period=20, std_dev=2.0)
    vol_avg = calculate_volume_avg(volume, 20)
    
    signals = np.zeros(n)
    
    # Base position sizing (Rule 4 - discrete levels, max 0.40)
    BASE_SIZE = 0.25
    MIN_SIZE = 0.20
    MAX_SIZE = 0.30
    
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
        
        if np.isnan(hma_4h_21_aligned[i]) or np.isnan(hma_1d_21_aligned[i]):
            continue
        
        if np.isnan(rsi_14[i]) or np.isnan(bb_upper[i]) or np.isnan(bb_lower[i]):
            continue
        
        # === HTF TREND BIAS (4h slope + 1d position) ===
        # More lenient: 4h slope positive OR price above 1d HMA for bullish
        htf_bullish = (slope_4h[i] > -0.5) or (close[i] > hma_1d_21_aligned[i])
        htf_bearish = (slope_4h[i] < 0.5) or (close[i] < hma_1d_21_aligned[i])
        
        # === SESSION FILTER (6-22 UTC - wider window) ===
        hour = get_session_hour(open_time[i])
        in_session = 6 <= hour <= 22
        
        # === VOLUME CONFIRMATION (lower threshold) ===
        volume_ok = volume[i] > 0.7 * vol_avg[i] if not np.isnan(vol_avg[i]) else True
        
        # === VOLATILITY SPIKE DETECTION ===
        vol_spike = (atr_7[i] / atr_30[i]) > 1.6 if not np.isnan(atr_30[i]) and atr_30[i] > 0 else False
        
        # === BOLLINGER BAND POSITION ===
        price_below_bb = close[i] < bb_lower[i]
        price_above_bb = close[i] > bb_upper[i]
        price_inside_bb = bb_lower[i] <= close[i] <= bb_upper[i]
        
        # === RSI EXTREMES (more lenient than CRSI) ===
        rsi_oversold = rsi_14[i] < 38
        rsi_overbought = rsi_14[i] > 62
        
        # === VOLATILITY-ADJUSTED POSITION SIZING ===
        atr_ratio = atr_14[i] / np.nanmedian(atr_14[max(0, i-100):i]) if i > 100 else 1.0
        vol_adjustment = np.clip(1.0 / atr_ratio, 0.8, 1.2)
        current_size = BASE_SIZE * vol_adjustment
        current_size = np.clip(current_size, MIN_SIZE, MAX_SIZE)
        current_size = np.round(current_size * 4) / 4  # Round to 0.05 increments
        current_size = np.clip(current_size, MIN_SIZE, MAX_SIZE)
        
        # === ENTRY LOGIC ===
        new_signal = 0.0
        bars_since_last_trade = i - last_trade_bar
        
        # LONG ENTRY: HTF bullish + (vol spike + BB lower OR RSI oversold) + session + volume
        long_condition_1 = htf_bullish and vol_spike and price_below_bb and in_session and volume_ok
        long_condition_2 = htf_bullish and rsi_oversold and price_below_bb and in_session
        long_condition_3 = htf_bullish and vol_spike and rsi_oversold and in_session and volume_ok
        
        if long_condition_1 or long_condition_2 or long_condition_3:
            new_signal = current_size
        
        # SHORT ENTRY: HTF bearish + (vol spike + BB upper OR RSI overbought) + session + volume
        short_condition_1 = htf_bearish and vol_spike and price_above_bb and in_session and volume_ok
        short_condition_2 = htf_bearish and rsi_overbought and price_above_bb and in_session
        short_condition_3 = htf_bearish and vol_spike and rsi_overbought and in_session and volume_ok
        
        if short_condition_1 or short_condition_2 or short_condition_3:
            new_signal = -current_size
        
        # === FREQUENCY SAFEGUARD ===
        # If no trades for 72 bars (~3 days on 1h), allow weaker entry
        if bars_since_last_trade > 72 and new_signal == 0.0 and not in_position:
            if htf_bullish and rsi_14[i] < 42 and price_below_bb and in_session:
                new_signal = current_size * 0.8
            elif htf_bearish and rsi_14[i] > 58 and price_above_bb and in_session:
                new_signal = -current_size * 0.8
        
        # === STOPLOSS LOGIC (Rule 6) - 2.5 * ATR ===
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
        
        # === MEAN REVERSION EXIT ===
        mean_rev_exit = False
        if in_position and position_side != 0:
            # Exit long when price crosses back above BB mid
            if position_side > 0 and close[i] > bb_mid[i]:
                mean_rev_exit = True
            # Exit short when price crosses back below BB mid
            if position_side < 0 and close[i] < bb_mid[i]:
                mean_rev_exit = True
        
        # === RSI NEUTRAL EXIT ===
        rsi_exit = False
        if in_position and position_side != 0:
            # Exit long when RSI becomes neutral/overbought
            if position_side > 0 and rsi_14[i] > 55:
                rsi_exit = True
            # Exit short when RSI becomes neutral/oversold
            if position_side < 0 and rsi_14[i] < 45:
                rsi_exit = True
        
        # === HTF TREND REVERSAL EXIT ===
        htf_reversal = False
        if in_position and position_side != 0:
            if position_side > 0 and (slope_4h[i] < -1.0 and close[i] < hma_1d_21_aligned[i]):
                htf_reversal = True
            if position_side < 0 and (slope_4h[i] > 1.0 and close[i] > hma_1d_21_aligned[i]):
                htf_reversal = True
        
        # Apply stoploss or reversals
        if stoploss_triggered or mean_rev_exit or rsi_exit or htf_reversal:
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