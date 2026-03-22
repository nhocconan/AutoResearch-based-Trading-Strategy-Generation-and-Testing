#!/usr/bin/env python3
"""
Experiment #098: 30m Primary + 4h/1d HTF — Triple TF Confluence with Session Filter

Hypothesis: Lower timeframe (30m) strategies fail due to too many trades and fee drag.
By using 1d for major trend direction, 4h for intermediate confirmation, and 30m only
for precise entry timing, we can achieve HTF trade frequency with LTF execution precision.

Strategy Logic:
1. 1d HMA(21) slope: Major trend bias (only long if slope > 0, only short if < 0)
2. 4h HMA(21) vs HMA(50): Intermediate trend confirmation
3. 30m RSI(14): Entry timing (only <25 for long, >75 for short)
4. Volume filter: volume > 0.8x 20-bar average
5. Session filter: Only trade 8-20 UTC (high liquidity hours)
6. ATR(14) stoploss: 2.5x trailing stop
7. Position size: 0.22 (smaller for lower TF to reduce fee impact)

Why this should work:
- 1d trend filter eliminates counter-trend trades (major source of losses)
- 4h confirmation adds confluence without over-filtering
- 30m RSI extremes catch pullbacks in trend
- Session filter reduces trades by ~60% (only 12/24 hours)
- Volume filter avoids low-liquidity fakeouts
- Small position size reduces drawdown and fee impact

Timeframe: 30m (REQUIRED)
HTF: 4h and 1d via mtf_data.get_htf_data() — called ONCE before loop
Position sizing: 0.22 discrete
Stoploss: 2.5 * ATR(14) trailing
Target trades: 40-80/year per symbol
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_30m_triple_tf_rsi_session_4h1d_v1"
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
    """Calculate HMA slope over lookback period as percentage."""
    slope = np.zeros(len(hma_values))
    for i in range(lookback, len(hma_values)):
        if hma_values[i - lookback] != 0 and not np.isnan(hma_values[i - lookback]):
            slope[i] = (hma_values[i] - hma_values[i - lookback]) / hma_values[i - lookback] * 100
        else:
            slope[i] = 0.0
    return slope

def calculate_volume_ratio(volume, period=20):
    """Calculate volume ratio vs rolling average."""
    vol_s = pd.Series(volume)
    vol_avg = vol_s.rolling(window=period, min_periods=period).mean().values
    vol_ratio = volume / vol_avg
    vol_ratio = np.nan_to_num(vol_ratio, nan=1.0)
    return vol_ratio

def extract_hour_from_open_time(prices):
    """Extract UTC hour from open_time column."""
    # open_time is typically in milliseconds or seconds since epoch
    # Convert to datetime and extract hour
    if prices['open_time'].dtype == 'int64':
        # Milliseconds since epoch
        hours = (prices['open_time'].values // (1000 * 60 * 60)) % 24
    else:
        # Already datetime
        hours = pd.to_datetime(prices['open_time'].values).hour
    return hours

def generate_signals(prices):
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    volume = prices["volume"].values
    n = len(close)
    
    # Extract session hours (Rule: only trade 8-20 UTC)
    hours = extract_hour_from_open_time(prices)
    
    # Load HTF data ONCE before loop (Rule 1 - CRITICAL)
    df_4h = get_htf_data(prices, '4h')
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate 1d indicators
    hma_1d_21 = calculate_hma(df_1d['close'].values, 21)
    hma_1d_slope = calculate_hma_slope(hma_1d_21, 5)
    
    # Calculate 4h indicators
    hma_4h_21 = calculate_hma(df_4h['close'].values, 21)
    hma_4h_50 = calculate_hma(df_4h['close'].values, 50)
    
    # Align HTF to LTF (Rule 2 - auto shift(1) for completed bars)
    hma_1d_21_aligned = align_htf_to_ltf(prices, df_1d, hma_1d_21)
    hma_1d_slope_aligned = align_htf_to_ltf(prices, df_1d, hma_1d_slope)
    hma_4h_21_aligned = align_htf_to_ltf(prices, df_4h, hma_4h_21)
    hma_4h_50_aligned = align_htf_to_ltf(prices, df_4h, hma_4h_50)
    
    # Calculate 30m indicators
    atr_14 = calculate_atr(high, low, close, 14)
    rsi_14 = calculate_rsi(close, 14)
    vol_ratio = calculate_volume_ratio(volume, 20)
    
    # 30m HMA for additional trend confirmation
    hma_30m_21 = calculate_hma(close, 21)
    hma_30m_50 = calculate_hma(close, 50)
    
    signals = np.zeros(n)
    
    # Base position sizing (Rule 4 - discrete levels, smaller for lower TF)
    BASE_SIZE = 0.22
    
    # Track position state for stoploss
    in_position = False
    position_side = 0
    entry_price = 0.0
    highest_price = 0.0
    lowest_price = 0.0
    last_trade_bar = -100
    
    for i in range(100, n):
        # Skip if indicators not ready
        if np.isnan(atr_14[i]) or atr_14[i] == 0:
            continue
        
        if np.isnan(hma_1d_21_aligned[i]) or np.isnan(hma_1d_slope_aligned[i]):
            continue
        
        if np.isnan(hma_4h_21_aligned[i]) or np.isnan(hma_4h_50_aligned[i]):
            continue
        
        if np.isnan(rsi_14[i]) or np.isnan(vol_ratio[i]):
            continue
        
        if np.isnan(hma_30m_21[i]) or np.isnan(hma_30m_50[i]):
            continue
        
        # === 1D TREND BIAS (MAJOR) ===
        # HMA slope > 0.3 = bullish bias (prefer longs)
        # HMA slope < -0.3 = bearish bias (prefer shorts)
        trend_1d_bullish = hma_1d_slope_aligned[i] > 0.3
        trend_1d_bearish = hma_1d_slope_aligned[i] < -0.3
        trend_1d_neutral = not trend_1d_bullish and not trend_1d_bearish
        
        # Price vs 1d HMA for additional confirmation
        price_above_1d_hma = close[i] > hma_1d_21_aligned[i]
        price_below_1d_hma = close[i] < hma_1d_21_aligned[i]
        
        # === 4H INTERMEDIATE TREND ===
        hma_4h_bullish = hma_4h_21_aligned[i] > hma_4h_50_aligned[i]
        hma_4h_bearish = hma_4h_21_aligned[i] < hma_4h_50_aligned[i]
        
        # === 30M TREND CONFIRMATION ===
        hma_30m_bullish = hma_30m_21[i] > hma_30m_50[i]
        hma_30m_bearish = hma_30m_21[i] < hma_30m_50[i]
        
        # === RSI EXTREMES FOR ENTRY ===
        rsi_oversold = rsi_14[i] < 25
        rsi_overbought = rsi_14[i] > 75
        rsi_neutral = 40 < rsi_14[i] < 60
        
        # === VOLUME FILTER ===
        volume_ok = vol_ratio[i] > 0.8
        
        # === SESSION FILTER (8-20 UTC) ===
        in_session = 8 <= hours[i] <= 20
        
        # === POSITION SIZING ===
        current_size = BASE_SIZE
        
        # === ENTRY LOGIC — VERY STRICT CONFLUENCE ===
        new_signal = 0.0
        bars_since_last_trade = i - last_trade_bar
        
        # LONG ENTRIES — require 4+ confluence factors
        long_confluence = 0
        if trend_1d_bullish:
            long_confluence += 2
        elif trend_1d_neutral and price_above_1d_hma:
            long_confluence += 1
        if hma_4h_bullish:
            long_confluence += 1
        if hma_30m_bullish:
            long_confluence += 1
        if rsi_oversold:
            long_confluence += 1
        if volume_ok:
            long_confluence += 0.5
        if in_session:
            long_confluence += 0.5
        
        # Need 4+ confluence for long entry
        if long_confluence >= 4.0 and in_session:
            new_signal = current_size
        
        # SHORT ENTRIES — require 4+ confluence factors
        short_confluence = 0
        if trend_1d_bearish:
            short_confluence += 2
        elif trend_1d_neutral and price_below_1d_hma:
            short_confluence += 1
        if hma_4h_bearish:
            short_confluence += 1
        if hma_30m_bearish:
            short_confluence += 1
        if rsi_overbought:
            short_confluence += 1
        if volume_ok:
            short_confluence += 0.5
        if in_session:
            short_confluence += 0.5
        
        # Need 4+ confluence for short entry
        if short_confluence >= 4.0 and in_session:
            new_signal = -current_size
        
        # === FREQUENCY SAFEGUARD ===
        # If no trades for 200 bars (~4 days on 30m), allow weaker entry (3.5 confluence)
        if bars_since_last_trade > 200 and new_signal == 0.0 and not in_position:
            if long_confluence >= 3.5 and in_session and trend_1d_bullish:
                new_signal = current_size * 0.7
            elif short_confluence >= 3.5 and in_session and trend_1d_bearish:
                new_signal = -current_size * 0.7
        
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
        # Exit if major trend reverses against position
        trend_reversal = False
        if in_position and position_side != 0:
            # Exit long if 1d trend turns bearish
            if position_side > 0 and trend_1d_bearish:
                trend_reversal = True
            # Exit short if 1d trend turns bullish
            if position_side < 0 and trend_1d_bullish:
                trend_reversal = True
        
        # Apply stoploss or trend reversal
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