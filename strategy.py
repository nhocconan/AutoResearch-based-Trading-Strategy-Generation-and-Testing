#!/usr/bin/env python3
"""
Experiment #018: 30m RSI Pullback with 4h/1d HMA Trend Confirmation

Hypothesis: Lower timeframe (30m) entries within higher timeframe (4h/1d) trend 
can capture more precise entry points while maintaining trend direction. This 
addresses the failure of pure mean-reversion (Connors/Choppiness) by using 
trend-following with pullback entries.

Key components:
1. 4h HMA(21) - primary trend direction (aligned with shift(1))
2. 1d HMA(48) - secular trend bias (aligned with shift(1))
3. 30m RSI(14) - entry timing on pullbacks (25-50 for long, 50-75 for short)
4. 30m ATR(14) - 2.5 ATR trailing stoploss
5. Volume filter - volume > 0.7x 20-bar average (relaxed for more trades)
6. Session filter - 8-20 UTC (high liquidity, but not mandatory)

Why this differs from failed #008 (mtf_30m_rsi_pullback_4h_1d_hma_session_vol_v1):
- Wider RSI range (25-50 / 50-75 instead of narrow bands)
- HMA instead of EMA for smoother trend
- Explicit stoploss tracking
- Discrete position sizing (0.20/0.22)
- Relaxed volume filter to ensure trade generation

Timeframe: 30m (REQUIRED)
HTF: 4h and 1d via mtf_data helper (call ONCE before loop)
Target trades: 40-80/year (balanced confluence to avoid fee drag)
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_30m_rsi_pullback_4h_1d_hma_vol_session_atr_v2"
timeframe = "30m"
leverage = 1.0

def calculate_rsi(close, period=14):
    """Calculate RSI using Wilder's smoothing."""
    close_s = pd.Series(close)
    delta = close_s.diff()
    gain = delta.where(delta > 0, 0.0)
    loss = -delta.where(delta < 0, 0.0)
    
    avg_gain = gain.ewm(span=period, min_periods=period, adjust=False).mean()
    avg_loss = loss.ewm(span=period, min_periods=period, adjust=False).mean()
    
    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))
    return rsi.values

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

def get_utc_hour(open_time):
    """Extract UTC hour from open_time (milliseconds timestamp)."""
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
    
    # Calculate 4h indicators
    hma_4h_21 = calculate_hma(df_4h['close'].values, 21)
    
    # Calculate 1d indicators
    hma_1d_48 = calculate_hma(df_1d['close'].values, 48)
    
    # Align HTF to LTF (Rule 2 - auto shift(1) for completed bars)
    hma_4h_21_aligned = align_htf_to_ltf(prices, df_4h, hma_4h_21)
    hma_1d_48_aligned = align_htf_to_ltf(prices, df_1d, hma_1d_48)
    
    # Calculate 30m indicators
    rsi_14 = calculate_rsi(close, 14)
    atr_14 = calculate_atr(high, low, close, 14)
    vol_avg_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    # Get UTC hours for session filter
    utc_hours = get_utc_hour(open_time)
    
    signals = np.zeros(n)
    
    # Position sizing (Rule 4 - discrete levels, max 0.40)
    BASE_SIZE_LONG = 0.22
    BASE_SIZE_SHORT = 0.20
    
    # Track position state for stoploss
    in_position = False
    position_side = 0
    entry_price = 0.0
    highest_price = 0.0
    lowest_price = 0.0
    
    for i in range(100, n):
        # Skip if indicators not ready
        if np.isnan(atr_14[i]) or atr_14[i] == 0:
            continue
        
        if np.isnan(hma_4h_21_aligned[i]) or np.isnan(hma_1d_48_aligned[i]):
            continue
        
        if np.isnan(rsi_14[i]):
            continue
        
        if np.isnan(vol_avg_20[i]) or vol_avg_20[i] == 0:
            continue
        
        # === HTF TREND DIRECTION ===
        trend_4h_bullish = close[i] > hma_4h_21_aligned[i]
        trend_4h_bearish = close[i] < hma_4h_21_aligned[i]
        
        trend_1d_bullish = close[i] > hma_1d_48_aligned[i]
        trend_1d_bearish = close[i] < hma_1d_48_aligned[i]
        
        # === VOLUME FILTER (relaxed for more trades) ===
        volume_ok = volume[i] > 0.7 * vol_avg_20[i]
        
        # === SESSION FILTER (8-20 UTC) - bonus filter, not mandatory ===
        session_ok = 8 <= utc_hours[i] <= 20
        
        # === RSI PULLBACK CONDITIONS (wider range for more trades) ===
        # Long: RSI pulled back to 25-50 in uptrend
        rsi_long_pullback = 25 <= rsi_14[i] <= 50
        
        # Short: RSI rallied to 50-75 in downtrend
        rsi_short_pullback = 50 <= rsi_14[i] <= 75
        
        # === ENTRY LOGIC ===
        new_signal = 0.0
        
        # LONG: 4h bullish + 1d bullish + RSI pullback + volume
        # Session is bonus (adds confidence but not required)
        long_core = trend_4h_bullish and trend_1d_bullish and rsi_long_pullback and volume_ok
        if long_core:
            new_signal = BASE_SIZE_LONG
        
        # SHORT: 4h bearish + 1d bearish + RSI pullback + volume
        short_core = trend_4h_bearish and trend_1d_bearish and rsi_short_pullback and volume_ok
        if short_core:
            new_signal = -BASE_SIZE_SHORT
        
        # === STOPLOSS LOGIC (Rule 6) - 2.5 * ATR trailing ===
        stoploss_triggered = False
        
        if in_position and position_side != 0:
            if position_side > 0:
                # Update highest price for long position
                if close[i] > highest_price:
                    highest_price = close[i]
                stoploss_price = highest_price - 2.5 * atr_14[i]
                if close[i] < stoploss_price:
                    stoploss_triggered = True
            
            if position_side < 0:
                # Update lowest price for short position
                if lowest_price == 0.0 or close[i] < lowest_price:
                    lowest_price = close[i]
                stoploss_price = lowest_price + 2.5 * atr_14[i]
                if close[i] > stoploss_price:
                    stoploss_triggered = True
        
        # === TREND REVERSAL EXIT ===
        trend_reversal = False
        if in_position and position_side != 0:
            # Exit long if 4h trend turns bearish
            if position_side > 0 and trend_4h_bearish:
                trend_reversal = True
            # Exit short if 4h trend turns bullish
            if position_side < 0 and trend_4h_bullish:
                trend_reversal = True
        
        # Apply stoploss or trend reversal
        if stoploss_triggered or trend_reversal:
            new_signal = 0.0
        
        # === UPDATE POSITION TRACKING ===
        if new_signal != 0.0:
            if not in_position:
                # New entry
                in_position = True
                position_side = np.sign(new_signal)
                entry_price = close[i]
                highest_price = close[i] if position_side > 0 else 0.0
                lowest_price = close[i] if position_side < 0 else 0.0
            elif np.sign(new_signal) != position_side:
                # Position flip
                position_side = np.sign(new_signal)
                entry_price = close[i]
                highest_price = close[i] if position_side > 0 else 0.0
                lowest_price = close[i] if position_side < 0 else 0.0
        else:
            if in_position:
                # Exit position
                in_position = False
                position_side = 0
                entry_price = 0.0
                highest_price = 0.0
                lowest_price = 0.0
        
        signals[i] = new_signal
    
    return signals