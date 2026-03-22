#!/usr/bin/env python3
"""
Experiment #068: 30m Primary + 4h/1d HTF — Volatility Breakout with Trend Filter

Hypothesis: Previous 30m strategies failed with 0 trades due to overly strict confluence.
This strategy SIMPLIFIES entry logic while maintaining quality filters:

1. 4h HMA(21) SLOPE for major trend direction (loose filter: just >0 or <0)
2. 30m RSI(14) momentum confirmation (threshold 45/55, not extreme 20/80)
3. 30m ATR(7)/ATR(30) ratio for volatility expansion (threshold >1.0, not >2.0)
4. Session filter 8-20 UTC (captures London/NY overlap, major volume)
5. Volume filter >0.6x 20-bar average (permissive, not 0.8x)
6. ATR(14) stoploss at 2.5x for 30m timeframe

Why this should work on 30m:
- 4h trend filter reduces false signals in counter-trend moves
- Loose RSI/ATR thresholds ensure trade generation (learned from #058, #060 failures)
- Session filter naturally limits trades to ~12 hours/day = fewer false breakouts
- Position size 0.20 accounts for higher frequency on 30m vs 12h/1d
- Target: 40-80 trades/year (within 30m limit of 50-100)

Timeframe: 30m (REQUIRED for this experiment)
HTF: 4h via mtf_data.get_htf_data() — called ONCE before loop
Position sizing: 0.20 discrete
Stoploss: 2.5 * ATR(14) trailing
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_30m_vol_breakout_4h_trend_session_v1"
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

def calculate_hma_slope(hma_values, lookback=3):
    """Calculate HMA slope over lookback period."""
    slope = np.zeros(len(hma_values))
    for i in range(lookback, len(hma_values)):
        if hma_values[i - lookback] != 0:
            slope[i] = (hma_values[i] - hma_values[i - lookback]) / hma_values[i - lookback] * 100
    return slope

def calculate_sma(close, period=200):
    """Calculate Simple Moving Average."""
    close_s = pd.Series(close)
    sma = close_s.rolling(window=period, min_periods=period).mean().values
    return sma

def extract_hour(open_time):
    """Extract UTC hour from open_time (milliseconds timestamp)."""
    # open_time is in milliseconds since epoch
    hours = (open_time // (1000 * 60 * 60)) % 24
    return hours

def generate_signals(prices):
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    open_time = prices["open_time"].values
    volume = prices["volume"].values
    n = len(close)
    
    # Load HTF data ONCE before loop (Rule 1 - CRITICAL)
    df_4h = get_htf_data(prices, '4h')
    
    # Calculate HTF indicators
    hma_4h_21 = calculate_hma(df_4h['close'].values, 21)
    hma_4h_slope = calculate_hma_slope(hma_4h_21, 3)
    
    # Align HTF to LTF (Rule 2 - auto shift(1) for completed bars)
    hma_4h_21_aligned = align_htf_to_ltf(prices, df_4h, hma_4h_21)
    hma_4h_slope_aligned = align_htf_to_ltf(prices, df_4h, hma_4h_slope)
    
    # Calculate 30m indicators
    atr_14 = calculate_atr(high, low, close, 14)
    atr_7 = calculate_atr(high, low, close, 7)
    atr_30 = calculate_atr(high, low, close, 30)
    rsi_14 = calculate_rsi(close, 14)
    sma_200 = calculate_sma(close, 200)
    
    # Volume moving average
    vol_sma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    # ATR ratio for volatility expansion
    atr_ratio = np.zeros(n)
    for i in range(30, n):
        if atr_30[i] > 0:
            atr_ratio[i] = atr_7[i] / atr_30[i]
    
    signals = np.zeros(n)
    
    # Base position sizing (Rule 4 - discrete levels, smaller for 30m)
    BASE_SIZE = 0.20
    
    # Track position state for stoploss
    in_position = False
    position_side = 0
    entry_price = 0.0
    highest_price = 0.0
    lowest_price = 0.0
    last_trade_bar = -100
    
    for i in range(200, n):
        # Skip if indicators not ready
        if np.isnan(atr_14[i]) or atr_14[i] == 0:
            continue
        
        if np.isnan(hma_4h_21_aligned[i]) or np.isnan(hma_4h_slope_aligned[i]):
            continue
        
        if np.isnan(rsi_14[i]) or np.isnan(sma_200[i]):
            continue
        
        if np.isnan(atr_ratio[i]) or np.isnan(vol_sma_20[i]):
            continue
        
        # Extract UTC hour for session filter
        hour = extract_hour(open_time[i])
        
        # === SESSION FILTER (8-20 UTC) ===
        # London open (8) to NY close (20) - major volume hours
        in_session = 8 <= hour <= 20
        
        # === VOLUME FILTER ===
        # Volume > 0.6x 20-bar average (permissive threshold)
        vol_ratio = volume[i] / vol_sma_20[i] if vol_sma_20[i] > 0 else 0
        volume_ok = vol_ratio > 0.6
        
        # === 4H TREND BIAS (MAJOR) ===
        # HMA slope > 0 = bullish bias (prefer longs)
        # HMA slope < 0 = bearish bias (prefer shorts)
        trend_4h_bullish = hma_4h_slope_aligned[i] > 0
        trend_4h_bearish = hma_4h_slope_aligned[i] < 0
        
        # Price vs 4h HMA for additional confirmation
        price_above_4h_hma = close[i] > hma_4h_21_aligned[i]
        price_below_4h_hma = close[i] < hma_4h_21_aligned[i]
        
        # Price vs 30m SMA200 for major trend
        price_above_sma200 = close[i] > sma_200[i]
        price_below_sma200 = close[i] < sma_200[i]
        
        # === 30M RSI MOMENTUM ===
        # RSI > 50 = bullish momentum
        # RSI < 50 = bearish momentum
        # Use 45/55 thresholds for entries (loose for trade generation)
        rsi_bullish = rsi_14[i] > 50
        rsi_bearish = rsi_14[i] < 50
        rsi_entry_long = rsi_14[i] > 45
        rsi_entry_short = rsi_14[i] < 55
        
        # === VOLATILITY EXPANSION ===
        # ATR(7)/ATR(30) > 1.0 = volatility expanding (loose threshold)
        vol_expanding = atr_ratio[i] > 1.0
        
        # === POSITION SIZING ===
        current_size = BASE_SIZE
        
        # Reduce size if volume is weak
        if vol_ratio < 1.0:
            current_size = BASE_SIZE * 0.7
        
        # === ENTRY LOGIC ===
        new_signal = 0.0
        bars_since_last_trade = i - last_trade_bar
        
        # LONG ENTRIES
        # Require: 4h bullish bias + RSI confirmation + session + volume
        # At least 2 of 3 trend confirmations (4h slope, 4h HMA, SMA200)
        trend_confirmations_long = sum([trend_4h_bullish, price_above_4h_hma, price_above_sma200])
        
        if trend_confirmations_long >= 2 and rsi_entry_long and in_session and volume_ok:
            # Strong entry with volatility expansion
            if vol_expanding and rsi_bullish:
                new_signal = current_size
            # Entry without vol expansion (still valid if trend strong)
            elif trend_confirmations_long >= 3 and rsi_14[i] > 50:
                new_signal = current_size * 0.8
        
        # SHORT ENTRIES
        # Require: 4h bearish bias + RSI confirmation + session + volume
        trend_confirmations_short = sum([trend_4h_bearish, price_below_4h_hma, price_below_sma200])
        
        if trend_confirmations_short >= 2 and rsi_entry_short and in_session and volume_ok:
            # Strong entry with volatility expansion
            if vol_expanding and rsi_bearish:
                new_signal = -current_size
            # Entry without vol expansion (still valid if trend strong)
            elif trend_confirmations_short >= 3 and rsi_14[i] < 50:
                new_signal = -current_size * 0.8
        
        # === FREQUENCY SAFEGUARD ===
        # If no trades for 200 bars (~4 days on 30m), allow weaker entry
        if bars_since_last_trade > 200 and new_signal == 0.0 and not in_position:
            if trend_4h_bullish and rsi_14[i] > 48 and in_session:
                new_signal = current_size * 0.5
            elif trend_4h_bearish and rsi_14[i] < 52 and in_session:
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
            # Exit long if 4h trend reverses bearish (2+ confirmations)
            if position_side > 0:
                trend_confirmations_bearish = sum([trend_4h_bearish, price_below_4h_hma, price_below_sma200])
                if trend_confirmations_bearish >= 2:
                    trend_reversal = True
            # Exit short if 4h trend reverses bullish (2+ confirmations)
            if position_side < 0:
                trend_confirmations_bullish = sum([trend_4h_bullish, price_above_4h_hma, price_above_sma200])
                if trend_confirmations_bullish >= 2:
                    trend_reversal = True
        
        # Apply stoploss or trend reversal
        if stoploss_triggered or trend_reversal:
            new_signal = 0.0
        
        # === SESSION EXIT ===
        # Close position if outside session hours (reduce overnight risk)
        if in_position and not in_session:
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