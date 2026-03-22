#!/usr/bin/env python3
"""
Experiment #010: 1h Multi-Timeframe Pullback Strategy with 4h/12h Trend Bias

Hypothesis: Lower timeframe (1h) can work IF we use HTF for direction and 1h only for entry timing.
Key insight from failures: regime-adaptive strategies overfit. Simple trend + pullback works better.

Why this should work:
1. 12h HMA provides major trend bias (like 1d but more responsive)
2. 4h HMA provides intermediate trend confirmation
3. 1h RSI pullback (30-50 for long, 50-70 for short) enters on retracements, not breakouts
4. Volume spike (1.5x avg) confirms institutional participation
5. Session filter (8-20 UTC) avoids low-liquidity Asian session whipsaws
6. ATR trailing stop (2.5x) protects from reversals
7. Very strict entry = fewer trades (target 40-60/year) = less fee drag

Position sizing: 0.25 discrete (smaller for 1h to account for more noise)
Stoploss: 2.5 * ATR(14) trailing
Trade frequency control: min 24 bars (24h) between entries same direction

Timeframe: 1h (REQUIRED for this experiment)
HTF: 4h and 12h via mtf_data helper (call ONCE before loop)
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_1h_pullback_4h12h_trend_volume_session_v1"
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

def calculate_volume_spike(volume, period=20):
    """Calculate volume ratio vs rolling average."""
    vol_s = pd.Series(volume)
    vol_avg = vol_s.rolling(window=period, min_periods=period).mean()
    vol_ratio = vol_s / vol_avg
    vol_ratio = vol_ratio.fillna(1.0).values
    return vol_ratio

def get_utc_hour(open_time):
    """Extract UTC hour from open_time (milliseconds timestamp)."""
    # open_time is in milliseconds
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
    df_12h = get_htf_data(prices, '12h')
    
    # Calculate HTF indicators
    hma_4h_21 = calculate_hma(df_4h['close'].values, 21)
    hma_12h_21 = calculate_hma(df_12h['close'].values, 21)
    
    # Align HTF to LTF (Rule 2 - auto shift(1) for completed bars)
    hma_4h_21_aligned = align_htf_to_ltf(prices, df_4h, hma_4h_21)
    hma_12h_21_aligned = align_htf_to_ltf(prices, df_12h, hma_12h_21)
    
    # Calculate 1h indicators
    atr_14 = calculate_atr(high, low, close, 14)
    rsi_14 = calculate_rsi(close, 14)
    vol_ratio = calculate_volume_spike(volume, 20)
    utc_hour = get_utc_hour(open_time)
    
    signals = np.zeros(n)
    
    # Base position sizing (Rule 4 - discrete levels, max 0.40)
    BASE_SIZE = 0.25
    
    # Track position state for stoploss
    in_position = False
    position_side = 0
    entry_price = 0.0
    highest_price = 0.0
    lowest_price = 0.0
    last_trade_bar = -100
    last_long_bar = -100
    last_short_bar = -100
    
    for i in range(100, n):
        # Skip if indicators not ready
        if np.isnan(atr_14[i]) or atr_14[i] == 0:
            continue
        
        if np.isnan(hma_4h_21_aligned[i]) or np.isnan(hma_12h_21_aligned[i]):
            continue
        
        if np.isnan(rsi_14[i]):
            continue
        
        # === HTF TREND BIAS (4h + 12h confluence) ===
        # Both 4h and 12h must agree for strong signal
        htf_bullish = (close[i] > hma_4h_21_aligned[i]) and (close[i] > hma_12h_21_aligned[i])
        htf_bearish = (close[i] < hma_4h_21_aligned[i]) and (close[i] < hma_12h_21_aligned[i])
        
        # === SESSION FILTER (8-20 UTC for liquidity) ===
        in_session = (utc_hour[i] >= 8) and (utc_hour[i] <= 20)
        
        # === VOLUME CONFIRMATION ===
        volume_confirmed = vol_ratio[i] > 1.2  # At least 20% above average
        
        # === RSI PULLBACK ENTRY (not extremes, but moderate pullbacks) ===
        # Long: RSI pulled back to 35-50 in uptrend
        rsi_long_pullback = (rsi_14[i] >= 35) and (rsi_14[i] <= 50)
        # Short: RSI pulled back to 50-65 in downtrend
        rsi_short_pullback = (rsi_14[i] >= 50) and (rsi_14[i] <= 65)
        
        # === FREQUENCY CONTROL ===
        bars_since_last_trade = i - last_trade_bar
        bars_since_last_long = i - last_long_bar
        bars_since_last_short = i - last_short_bar
        
        # Minimum 24 bars (24h) between entries same direction
        can_long = bars_since_last_long >= 24
        can_short = bars_since_last_short >= 24
        
        # === ENTRY LOGIC (3+ confluence required) ===
        new_signal = 0.0
        
        # LONG: HTF bullish + RSI pullback + volume + session + frequency ok
        if htf_bullish and rsi_long_pullback and volume_confirmed and in_session and can_long:
            new_signal = BASE_SIZE
        
        # SHORT: HTF bearish + RSI pullback + volume + session + frequency ok
        elif htf_bearish and rsi_short_pullback and volume_confirmed and in_session and can_short:
            new_signal = -BASE_SIZE
        
        # === STOPLOSS LOGIC (Rule 6) - 2.5 * ATR trailing ===
        stoploss_triggered = False
        
        if in_position and position_side != 0:
            if position_side > 0:
                # Update highest price for long
                if close[i] > highest_price:
                    highest_price = close[i]
                stoploss_price = highest_price - 2.5 * atr_14[i]
                if close[i] < stoploss_price:
                    stoploss_triggered = True
            
            if position_side < 0:
                # Update lowest price for short
                if lowest_price == 0.0 or close[i] < lowest_price:
                    lowest_price = close[i]
                stoploss_price = lowest_price + 2.5 * atr_14[i]
                if close[i] > stoploss_price:
                    stoploss_triggered = True
        
        # === TREND REVERSAL EXIT ===
        trend_reversal = False
        if in_position and position_side != 0:
            if position_side > 0 and htf_bearish:
                trend_reversal = True
            if position_side < 0 and htf_bullish:
                trend_reversal = True
        
        # === RSI EXTREME EXIT (take profit) ===
        rsi_exit = False
        if in_position and position_side != 0:
            if position_side > 0 and rsi_14[i] > 70:
                rsi_exit = True
            if position_side < 0 and rsi_14[i] < 30:
                rsi_exit = True
        
        # Apply stoploss or trend reversal or RSI exit
        if stoploss_triggered or trend_reversal or rsi_exit:
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
                if position_side > 0:
                    last_long_bar = i
                else:
                    last_short_bar = i
            elif np.sign(new_signal) != position_side:
                # Flip position
                position_side = np.sign(new_signal)
                entry_price = close[i]
                highest_price = close[i] if position_side > 0 else 0.0
                lowest_price = close[i] if position_side < 0 else 0.0
                last_trade_bar = i
                if position_side > 0:
                    last_long_bar = i
                else:
                    last_short_bar = i
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