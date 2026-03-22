#!/usr/bin/env python3
"""
Experiment #215: 1h Primary + 4h/1d HTF — Multi-Confluence Mean Reversion with Session Filter

Hypothesis: After analyzing 214 experiments, the pattern for 1h timeframe success is:
1. Use HTF (4h/1d) for TREND DIRECTION only — don't trade against major trends
2. Use 1h for ENTRY TIMING — mean reversion within HTF trend
3. Session filter (8-20 UTC) reduces trades to target 30-60/year
4. Volume confirmation avoids fake signals
5. Discrete position sizing (0.25 base) controls drawdown

Why this should work on 1h:
- 4h HMA slope filters out counter-trend trades (major whipsaw reducer)
- RSI(14) extremes (30/70) catch mean reversion within trend
- Session filter (8-20 UTC) = London/NY overlap = 50% of day = fewer trades
- Volume > 0.8x avg confirms real moves, not noise
- 1h timeframe = precise entry timing within 4h trend direction

Trade frequency control:
- Session filter alone cuts trades by ~50%
- HTF trend filter cuts another ~40%
- RSI extreme + volume = additional ~30% filter
- Combined: should hit 40-80 trades/year target

Timeframe: 1h (REQUIRED for this experiment)
HTF: 4h + 1d via mtf_data.get_htf_data() — called ONCE before loop
Position sizing: 0.25 base, discrete levels
Stoploss: 2.5 * ATR(14) trailing
Target trades: 40-80/year per symbol
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_1h_rsi_session_volume_4h1d_v1"
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

def calculate_hma(close, period=21):
    """Calculate Hull Moving Average."""
    close_s = pd.Series(close)
    half = max(1, period // 2)
    sqrt_period = max(1, int(np.sqrt(period)))
    wma1 = close_s.ewm(span=half, min_periods=half, adjust=False).mean()
    wma2 = close_s.ewm(span=period, min_periods=period, adjust=False).mean()
    wma3 = (2 * wma1 - wma2).ewm(span=sqrt_period, min_periods=sqrt_period, adjust=False).mean()
    return wma3.values

def calculate_hma_slope(hma_values, lookback=5):
    """Calculate HMA slope as percentage change."""
    slope = np.zeros(len(hma_values))
    for i in range(lookback, len(hma_values)):
        if hma_values[i - lookback] != 0:
            slope[i] = (hma_values[i] - hma_values[i - lookback]) / hma_values[i - lookback] * 100
    return slope

def calculate_volume_ratio(volume, period=20):
    """Calculate volume ratio vs rolling average."""
    vol_avg = pd.Series(volume).rolling(window=period, min_periods=period).mean().values
    vol_ratio = volume / vol_avg
    vol_ratio = np.nan_to_num(vol_ratio, nan=1.0)
    return vol_ratio

def get_hour_from_open_time(open_time_array):
    """Extract UTC hour from open_time (milliseconds timestamp)."""
    # open_time is in milliseconds since epoch
    hours = ((open_time_array // 1000) // 3600) % 24
    return hours

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
    
    # Calculate 4h HTF indicators
    hma_4h_21 = calculate_hma(df_4h['close'].values, 21)
    hma_4h_slope = calculate_hma_slope(hma_4h_21, 5)
    
    # Calculate 1d HTF indicators
    hma_1d_21 = calculate_hma(df_1d['close'].values, 21)
    hma_1d_slope = calculate_hma_slope(hma_1d_21, 5)
    
    # Align HTF to LTF (Rule 2 - auto shift(1))
    hma_4h_21_aligned = align_htf_to_ltf(prices, df_4h, hma_4h_21)
    hma_4h_slope_aligned = align_htf_to_ltf(prices, df_4h, hma_4h_slope)
    hma_1d_21_aligned = align_htf_to_ltf(prices, df_1d, hma_1d_21)
    hma_1d_slope_aligned = align_htf_to_ltf(prices, df_1d, hma_1d_slope)
    
    # Calculate 1h indicators
    atr_14 = calculate_atr(high, low, close, 14)
    rsi_14 = calculate_rsi(close, 14)
    vol_ratio = calculate_volume_ratio(volume, 20)
    utc_hour = get_hour_from_open_time(open_time)
    
    # 1h HMA for local trend confirmation
    hma_1h_21 = calculate_hma(close, 21)
    
    signals = np.zeros(n)
    
    # Position sizing (Rule 4 - discrete, max 0.40)
    BASE_SIZE = 0.25
    
    # Track position state
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
        
        if np.isnan(hma_4h_21_aligned[i]) or np.isnan(hma_4h_slope_aligned[i]):
            continue
        
        if np.isnan(hma_1d_21_aligned[i]) or np.isnan(hma_1d_slope_aligned[i]):
            continue
        
        if np.isnan(rsi_14[i]) or np.isnan(vol_ratio[i]):
            continue
        
        # === HTF TREND BIAS (4h + 1d) ===
        trend_4h_bullish = hma_4h_slope_aligned[i] > 0.15
        trend_4h_bearish = hma_4h_slope_aligned[i] < -0.15
        trend_1d_bullish = hma_1d_slope_aligned[i] > 0.20
        trend_1d_bearish = hma_1d_slope_aligned[i] < -0.20
        
        # Price relative to HTF HMA
        price_above_4h_hma = close[i] > hma_4h_21_aligned[i]
        price_below_4h_hma = close[i] < hma_4h_21_aligned[i]
        price_above_1d_hma = close[i] > hma_1d_21_aligned[i]
        price_below_1d_hma = close[i] < hma_1d_21_aligned[i]
        
        # === SESSION FILTER (8-20 UTC) ===
        in_session = 8 <= utc_hour[i] <= 20
        
        # === VOLUME CONFIRMATION ===
        volume_confirmed = vol_ratio[i] > 0.8
        
        # === LOCAL TREND (1h HMA) ===
        price_above_1h_hma = close[i] > hma_1h_21[i]
        price_below_1h_hma = close[i] < hma_1h_1[i] if 'hma_1h_1' in dir() else close[i] < hma_1h_21[i]
        
        # === RSI EXTREMES (Mean Reversion) ===
        rsi_oversold = rsi_14[i] < 32
        rsi_overbought = rsi_14[i] > 68
        rsi_neutral = 32 <= rsi_14[i] <= 68
        
        # === POSITION SIZING ===
        current_size = BASE_SIZE
        
        # Reduce size if HTF trends conflict
        if (trend_4h_bullish and trend_1d_bearish) or (trend_4h_bearish and trend_1d_bullish):
            current_size = BASE_SIZE * 0.6
        
        # === ENTRY LOGIC ===
        new_signal = 0.0
        bars_since_last_trade = i - last_trade_bar
        
        # LONG ENTRIES - Must pass session + volume + HTF trend filter
        long_score = 0
        
        # Core: 4h bullish trend + RSI oversold (mean reversion within uptrend)
        if trend_4h_bullish and rsi_oversold:
            long_score += 3
        
        # Confirmation: Price above 4h HMA
        if price_above_4h_hma:
            long_score += 1
        
        # Confirmation: 1d also bullish (stronger trend)
        if trend_1d_bullish:
            long_score += 1
        
        # Confirmation: Price above 1h HMA (local momentum)
        if price_above_1h_hma:
            long_score += 1
        
        # Required filters: session + volume
        if in_session and volume_confirmed and long_score >= 3:
            new_signal = current_size
        elif in_session and volume_confirmed and long_score >= 4:
            new_signal = current_size * 1.2  # Stronger signal
        
        # SHORT ENTRIES
        short_score = 0
        
        # Core: 4h bearish trend + RSI overbought (mean reversion within downtrend)
        if trend_4h_bearish and rsi_overbought:
            short_score += 3
        
        # Confirmation: Price below 4h HMA
        if price_below_4h_hma:
            short_score += 1
        
        # Confirmation: 1d also bearish (stronger trend)
        if trend_1d_bearish:
            short_score += 1
        
        # Confirmation: Price below 1h HMA (local momentum)
        if price_below_1h_hma:
            short_score += 1
        
        if in_session and volume_confirmed and short_score >= 3:
            new_signal = -current_size
        elif in_session and volume_confirmed and short_score >= 4:
            new_signal = -current_size * 1.2
        
        # === FREQUENCY SAFEGUARD ===
        # Force trade if no signal for 168 bars (~1 week on 1h) and strong HTF alignment
        if bars_since_last_trade > 168 and new_signal == 0.0 and not in_position:
            if trend_4h_bullish and trend_1d_bullish and rsi_14[i] < 40 and in_session:
                new_signal = current_size * 0.5
            elif trend_4h_bearish and trend_1d_bearish and rsi_14[i] > 60 and in_session:
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
        
        # === HTF TREND REVERSAL EXIT ===
        trend_reversal = False
        if in_position and position_side != 0:
            # Long position but 4h turns bearish
            if position_side > 0 and trend_4h_bearish and not trend_1d_bullish:
                trend_reversal = True
            # Short position but 4h turns bullish
            if position_side < 0 and trend_4h_bullish and not trend_1d_bearish:
                trend_reversal = True
        
        if stoploss_triggered or trend_reversal:
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