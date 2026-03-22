#!/usr/bin/env python3
"""
Experiment #018: 30m Multi-Timeframe HMA + RSI Pullback with Session/Volume Filter

Hypothesis: 30m strategies fail due to (1) too many trades → fee drag, or (2) too strict filters → 0 trades.
Solution: Use 4h/1d for TREND DIRECTION (fewer signals), 30m only for ENTRY TIMING.
Add session filter (8-20 UTC) to avoid low-liquidity whipsaws. Volume filter confirms real moves.
RSI pullback (30-70, NOT extreme 20/80) catches more entries than oversold/overbought extremes.

Why this should work:
- 4h HMA(16/48) crossover = proven trend filter (from best baseline mtf_4h_supertrend_hma_1d_rsi_adx_v1)
- 1d HMA(21) = major trend bias (prevents counter-trend trades in bear markets)
- 30m RSI(14) 30-70 = pullback entries (MORE frequent than 20/80 extremes that caused 0 trades)
- Session 8-20 UTC = avoids Asian low-liquidity whipsaws (reduces false signals)
- Volume > 0.7x avg = confirms real moves (not too strict)
- ATR 2.5x stop = protects from crashes like 2022 -77% drop
- Size 0.25 = smaller for lower TF (less fee impact per trade)

CRITICAL: Entry conditions LOOSENED vs failed experiments (#008, #010, #015 had 0 trades)
- RSI 30-70 instead of 20/80 extremes
- Volume 0.7x instead of 1.5x
- 4h trend PRIMARY, 1d bias SECONDARY (not both required equally)
- Session filter can be bypassed for very strong signals

Timeframe: 30m (REQUIRED for this experiment)
HTF: 4h and 1d via mtf_data helper (call ONCE before loop!)
Position sizing: 0.25 discrete (smaller for 30m to reduce fee drag)
Stoploss: 2.5 * ATR(14) trailing
Target trades: 40-80/year (NOT >100 which kills profit via fees)
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf
from datetime import datetime, timezone

name = "mtf_30m_hma_rsi_session_vol_4h_1d_v1"
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

def calculate_volume_sma(volume, period=20):
    """Calculate simple moving average of volume."""
    vol_s = pd.Series(volume)
    vol_sma = vol_s.rolling(window=period, min_periods=period).mean().values
    return vol_sma

def get_utc_hour(timestamp_ms):
    """Extract UTC hour from millisecond timestamp."""
    timestamp_s = timestamp_ms / 1000.0
    dt = datetime.fromtimestamp(timestamp_s, tz=timezone.utc)
    return dt.hour

def generate_signals(prices):
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    volume = prices["volume"].values
    n = len(close)
    
    # Load HTF data ONCE before loop (Rule 1 - CRITICAL)
    # This is the #1 mistake - calling get_htf_data inside loop = 45K file reads = HANG
    df_4h = get_htf_data(prices, '4h')
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate HTF indicators
    hma_4h_16 = calculate_hma(df_4h['close'].values, 16)
    hma_4h_48 = calculate_hma(df_4h['close'].values, 48)
    hma_1d_21 = calculate_hma(df_1d['close'].values, 21)
    
    # Align HTF to LTF (Rule 2 - auto shift(1) for completed bars only)
    # This prevents look-ahead - uses PREVIOUS completed HTF bar
    hma_4h_16_aligned = align_htf_to_ltf(prices, df_4h, hma_4h_16)
    hma_4h_48_aligned = align_htf_to_ltf(prices, df_4h, hma_4h_48)
    hma_1d_21_aligned = align_htf_to_ltf(prices, df_1d, hma_1d_21)
    
    # Calculate 30m indicators
    atr_14 = calculate_atr(high, low, close, 14)
    rsi_14 = calculate_rsi(close, 14)
    vol_sma_20 = calculate_volume_sma(volume, 20)
    
    signals = np.zeros(n)
    
    # Position sizing (Rule 4 - discrete levels, max 0.40, smaller for lower TF)
    BASE_SIZE = 0.25  # 25% of capital (smaller than 12h/4h strategies)
    
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
        
        if np.isnan(hma_4h_16_aligned[i]) or np.isnan(hma_4h_48_aligned[i]):
            continue
        
        if np.isnan(hma_1d_21_aligned[i]):
            continue
        
        if np.isnan(rsi_14[i]):
            continue
        
        if np.isnan(vol_sma_20[i]) or vol_sma_20[i] == 0:
            continue
        
        # === GET UTC HOUR FOR SESSION FILTER ===
        hour = get_utc_hour(prices['open_time'].iloc[i])
        in_session = 8 <= hour <= 20  # London/NY overlap = high liquidity
        
        # === VOLUME CONFIRMATION ===
        volume_ratio = volume[i] / vol_sma_20[i] if vol_sma_20[i] > 0 else 1.0
        volume_confirmed = volume_ratio > 0.70  # Not too strict (0.7x not 1.5x)
        
        # === 4H TREND (PRIMARY SIGNAL) ===
        hma_4h_bullish = hma_4h_16_aligned[i] > hma_4h_48_aligned[i]
        hma_4h_bearish = hma_4h_16_aligned[i] < hma_4h_48_aligned[i]
        
        # === 1D BIAS (SECONDARY CONFIRMATION) ===
        daily_bullish = close[i] > hma_1d_21_aligned[i]
        daily_bearish = close[i] < hma_1d_21_aligned[i]
        
        # === ENTRY LOGIC (LOOSENED vs failed experiments) ===
        new_signal = 0.0
        bars_since_last_trade = i - last_trade_bar
        
        # LONG: 4h HMA bullish + (1d bias bullish OR strong 4h signal) + RSI pullback
        if hma_4h_bullish:
            # RSI pullback to 30-60 range (LOOSENED from 20/80 extremes)
            rsi_long = 30 <= rsi_14[i] <= 60
            
            # Strong signal: both 4h and 1d agree (can bypass session filter)
            strong_long = hma_4h_bullish and daily_bullish and rsi_long and volume_confirmed
            
            # Normal signal: 4h bullish + session + volume + RSI
            normal_long = hma_4h_bullish and in_session and volume_confirmed and rsi_long
            
            if strong_long or normal_long:
                new_signal = BASE_SIZE
        
        # SHORT: 4h HMA bearish + (1d bias bearish OR strong 4h signal) + RSI pullback
        elif hma_4h_bearish:
            # RSI pullback to 40-70 range (LOOSENED from 20/80 extremes)
            rsi_short = 40 <= rsi_14[i] <= 70
            
            # Strong signal: both 4h and 1d agree (can bypass session filter)
            strong_short = hma_4h_bearish and daily_bearish and rsi_short and volume_confirmed
            
            # Normal signal: 4h bearish + session + volume + RSI
            normal_short = hma_4h_bearish and in_session and volume_confirmed and rsi_short
            
            if strong_short or normal_short:
                new_signal = -BASE_SIZE
        
        # === FREQUENCY SAFEGUARD (prevent 0 trades) ===
        # If no trades for 60 bars (~30 hours on 30m), force entry with weaker conditions
        if bars_since_last_trade > 60 and new_signal == 0.0 and not in_position:
            if hma_4h_bullish and daily_bullish and 35 <= rsi_14[i] <= 55:
                new_signal = BASE_SIZE * 0.6  # Reduced size for forced entry
            elif hma_4h_bearish and daily_bearish and 45 <= rsi_14[i] <= 65:
                new_signal = -BASE_SIZE * 0.6
        
        # === STOPLOSS LOGIC (Rule 6) - 2.5 * ATR trailing ===
        stoploss_triggered = False
        
        if in_position and position_side != 0:
            if position_side > 0:
                # Long: track highest price, stop below
                if close[i] > highest_price:
                    highest_price = close[i]
                stoploss_price = highest_price - 2.5 * atr_14[i]
                if close[i] < stoploss_price:
                    stoploss_triggered = True
            
            if position_side < 0:
                # Short: track lowest price, stop above
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
                # Flip position
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