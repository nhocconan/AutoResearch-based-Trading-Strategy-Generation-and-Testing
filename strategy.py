#!/usr/bin/env python3
"""
Experiment #028: 30m Low-Frequency HTF-Guided Entry with Strict Confluence

Hypothesis: Lower TF (30m) strategies fail due to excessive trade frequency → fee drag.
Solution: Use 4h/1d for TREND DIRECTION, 30m ONLY for rare entry timing.
Require 4+ confluence filters to trigger entry (target 30-60 trades/year).

Key filters:
1. 4h HMA(16/48) crossover for trend direction
2. 1d HMA(21) for major bias (price above/below)
3. 30m RSI(14) pullback to 35-45 (long) or 55-65 (short) - NOT extremes
4. Volume spike > 1.2x 20-bar average (confirms momentum)
5. Session filter: 8-20 UTC only (avoid low-liquidity hours)
6. ADX(14) > 22 on 4h (confirms trend strength, avoids chop)
7. Price vs 30m EMA(21) for entry timing precision

Position sizing: 0.22 (smaller for lower TF risk)
Stoploss: 2.5 * ATR(14) trailing
Timeframe: 30m (REQUIRED)
HTF: 4h and 1d via mtf_data helper (call ONCE before loop)

Why this differs from failed #018:
- Stricter confluence (4+ filters vs 2-3)
- Volume confirmation required
- Session filter reduces low-liquidity trades
- ADX filter avoids choppy 4h conditions
- Smaller position size (0.22 vs 0.30)
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_30m_htf_confluence_volume_session_v1"
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

def calculate_hma(close, period=21):
    """Calculate Hull Moving Average for smoother trend with less lag."""
    close_s = pd.Series(close)
    half = max(1, period // 2)
    sqrt_period = max(1, int(np.sqrt(period)))
    wma1 = close_s.ewm(span=half, min_periods=half, adjust=False).mean()
    wma2 = close_s.ewm(span=period, min_periods=period, adjust=False).mean()
    wma3 = (2 * wma1 - wma2).ewm(span=sqrt_period, min_periods=sqrt_period, adjust=False).mean()
    return wma3.values

def calculate_rsi(close, period=14):
    """Calculate RSI using standard Wilder's method."""
    close_s = pd.Series(close)
    delta = close_s.diff()
    gain = delta.where(delta > 0, 0.0)
    loss = -delta.where(delta < 0, 0.0)
    
    avg_gain = gain.ewm(span=period, min_periods=period, adjust=False).mean()
    avg_loss = loss.ewm(span=period, min_periods=period, adjust=False).mean()
    
    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))
    rsi = rsi.fillna(50).values
    return rsi

def calculate_adx(high, low, close, period=14):
    """Calculate ADX (Average Directional Index) for trend strength."""
    high_s = pd.Series(high)
    low_s = pd.Series(low)
    close_s = pd.Series(close)
    
    plus_dm = high_s.diff()
    minus_dm = -low_s.diff()
    
    plus_dm = plus_dm.where((plus_dm > minus_dm) & (plus_dm > 0), 0.0)
    minus_dm = minus_dm.where((minus_dm > plus_dm) & (minus_dm > 0), 0.0)
    
    tr1 = high_s - low_s
    tr2 = np.abs(high_s - close_s.shift(1))
    tr3 = np.abs(low_s - close_s.shift(1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    
    atr = tr.ewm(span=period, min_periods=period, adjust=False).mean()
    plus_di = 100 * (plus_dm.ewm(span=period, min_periods=period, adjust=False).mean() / atr)
    minus_di = 100 * (minus_dm.ewm(span=period, min_periods=period, adjust=False).mean() / atr)
    
    dx = 100 * np.abs(plus_di - minus_di) / (plus_di + minus_di + 1e-10)
    adx = dx.ewm(span=period, min_periods=period, adjust=False).mean()
    
    return adx.values

def calculate_ema(close, period=21):
    """Calculate Exponential Moving Average."""
    close_s = pd.Series(close)
    ema = close_s.ewm(span=period, min_periods=period, adjust=False).mean()
    return ema.values

def extract_hour_from_open_time(prices):
    """Extract UTC hour from open_time column."""
    # open_time is in milliseconds since epoch
    hours = (prices['open_time'].values // (1000 * 60 * 60)) % 24
    return hours

def generate_signals(prices):
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    volume = prices["volume"].values
    n = len(close)
    
    # Load HTF data ONCE before loop (Rule 1 - CRITICAL)
    df_4h = get_htf_data(prices, '4h')
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate 4H indicators
    hma_4h_16 = calculate_hma(df_4h['close'].values, 16)
    hma_4h_48 = calculate_hma(df_4h['close'].values, 48)
    adx_4h_14 = calculate_adx(df_4h['high'].values, df_4h['low'].values, df_4h['close'].values, 14)
    
    # Calculate 1D indicators
    hma_1d_21 = calculate_hma(df_1d['close'].values, 21)
    
    # Align HTF to LTF (Rule 2 - auto shift(1) for completed bars)
    hma_4h_16_aligned = align_htf_to_ltf(prices, df_4h, hma_4h_16)
    hma_4h_48_aligned = align_htf_to_ltf(prices, df_4h, hma_4h_48)
    adx_4h_14_aligned = align_htf_to_ltf(prices, df_4h, adx_4h_14)
    hma_1d_21_aligned = align_htf_to_ltf(prices, df_1d, hma_1d_21)
    
    # Calculate 30m indicators
    atr_14 = calculate_atr(high, low, close, 14)
    rsi_14 = calculate_rsi(close, 14)
    ema_30m_21 = calculate_ema(close, 21)
    
    # Volume moving average (20 bars)
    vol_s = pd.Series(volume)
    vol_ma_20 = vol_s.rolling(window=20, min_periods=20).mean().values
    
    # Extract UTC hour for session filter
    hours = extract_hour_from_open_time(prices)
    
    signals = np.zeros(n)
    
    # Base position sizing (Rule 4 - discrete levels, smaller for 30m)
    BASE_SIZE = 0.22
    
    # Track position state for stoploss
    in_position = False
    position_side = 0
    entry_price = 0.0
    highest_price = 0.0
    lowest_price = 0.0
    last_trade_bar = -100
    
    for i in range(150, n):
        # Skip if indicators not ready
        if np.isnan(atr_14[i]) or atr_14[i] == 0:
            continue
        
        if np.isnan(hma_4h_16_aligned[i]) or np.isnan(hma_4h_48_aligned[i]):
            continue
        
        if np.isnan(hma_1d_21_aligned[i]):
            continue
        
        if np.isnan(adx_4h_14_aligned[i]):
            continue
        
        if np.isnan(rsi_14[i]):
            continue
        
        if np.isnan(vol_ma_20[i]) or vol_ma_20[i] == 0:
            continue
        
        # === 4H TREND DIRECTION ===
        hma_4h_bullish = hma_4h_16_aligned[i] > hma_4h_48_aligned[i]
        hma_4h_bearish = hma_4h_16_aligned[i] < hma_4h_48_aligned[i]
        
        # === 1D BIAS FILTER ===
        daily_bullish = close[i] > hma_1d_21_aligned[i]
        daily_bearish = close[i] < hma_1d_21_aligned[i]
        
        # === 4H TREND STRENGTH (ADX) ===
        adx_strong = adx_4h_14_aligned[i] > 22
        
        # === SESSION FILTER (8-20 UTC) ===
        in_session = 8 <= hours[i] <= 20
        
        # === VOLUME SPIKE ===
        volume_ratio = volume[i] / vol_ma_20[i]
        volume_spike = volume_ratio > 1.15
        
        # === 30M RSI PULLBACK ===
        rsi_long_pullback = 35 <= rsi_14[i] <= 50
        rsi_short_pullback = 50 <= rsi_14[i] <= 65
        
        # === 30M PRICE VS EMA ===
        price_above_ema = close[i] > ema_30m_21[i]
        price_below_ema = close[i] < ema_30m_21[i]
        
        # === VOLATILITY-ADJUSTED POSITION SIZING ===
        if i > 150:
            atr_median = np.nanmedian(atr_14[max(0, i-100):i])
            atr_ratio = atr_14[i] / atr_median if atr_median > 0 else 1.0
            vol_adjustment = np.clip(1.0 / atr_ratio, 0.7, 1.3)
        else:
            vol_adjustment = 1.0
        
        current_size = BASE_SIZE * vol_adjustment
        current_size = np.clip(current_size, 0.18, 0.28)
        
        # === ENTRY LOGIC (STRICT CONFLUENCE) ===
        new_signal = 0.0
        bars_since_last_trade = i - last_trade_bar
        
        # LONG: 4H bullish + 1D bullish + ADX strong + session + volume + RSI pullback + price>EMA
        if hma_4h_bullish and daily_bullish and adx_strong:
            if in_session and volume_spike and rsi_long_pullback and price_above_ema:
                new_signal = current_size
        
        # SHORT: 4H bearish + 1D bearish + ADX strong + session + volume + RSI pullback + price<EMA
        elif hma_4h_bearish and daily_bearish and adx_strong:
            if in_session and volume_spike and rsi_short_pullback and price_below_ema:
                new_signal = -current_size
        
        # === FREQUENCY SAFEGUARD ===
        # If no trades for 80 bars (~40 hours on 30m), allow weaker entry
        if bars_since_last_trade > 80 and new_signal == 0.0 and not in_position:
            if hma_4h_bullish and daily_bullish and rsi_14[i] > 40 and rsi_14[i] < 55:
                new_signal = current_size * 0.5
            elif hma_4h_bearish and daily_bearish and rsi_14[i] > 45 and rsi_14[i] < 60:
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
        
        # === TREND REVERSAL EXIT ===
        trend_reversal = False
        if in_position and position_side != 0:
            if position_side > 0 and hma_4h_bearish:
                trend_reversal = True
            if position_side < 0 and hma_4h_bullish:
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