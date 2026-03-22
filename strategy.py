#!/usr/bin/env python3
"""
Experiment #035: 1h HMA Trend + RSI Pullback with 4h/1d Confirmation

Hypothesis: Previous 1h strategies failed due to over-filtering (too many confluence 
requirements → 0 trades). This simplifies to core trend-following with pullback entries:
1. 4h HMA(21) sets primary trend direction (call ONCE before loop)
2. 1d HMA(21) confirms macro bias (call ONCE before loop)
3. 1h RSI(14) pullback within trend (RSI 35-55 for long, 45-65 for short)
4. Volume confirmation (>0.8x 20-bar avg)
5. ATR(14) trailing stop at 2.5x

Key differences from failed #025/#030:
- Removed Choppiness Index (over-filtering → 0 trades)
- Removed Connors RSI (too complex, same issue)
- Simplified RSI to standard 14-period with pullback ranges
- Reduced confluence from 5+ filters to 3 core (HTF trend + RSI + volume)
- Position size 0.25 (conservative for 1h TF)

Why this should work on 1h:
- 4h/1d HTF provides direction (reduces whipsaw)
- 1h RSI pullback gives precise entry timing
- Volume filter avoids low-liquidity traps
- Target: 40-60 trades/year (fee drag manageable at 0.05% RT)

Timeframe: 1h (REQUIRED)
HTF: 4h and 1d via mtf_data helper (call ONCE before loop)
Position sizing: 0.25 discrete
Stoploss: 2.5 * ATR(14) trailing
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_1h_hma_trend_rsi_pullback_4h_1d_vol_atr_v1"
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
    return rsi.values

def calculate_sma(values, period=20):
    """Calculate Simple Moving Average."""
    return pd.Series(values).rolling(window=period, min_periods=period).mean().values

def generate_signals(prices):
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    volume = prices["volume"].values
    n = len(close)
    
    # Load HTF data ONCE before loop (Rule 1 - CRITICAL)
    df_4h = get_htf_data(prices, '4h')
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate HTF indicators
    hma_4h_21 = calculate_hma(df_4h['close'].values, 21)
    hma_1d_21 = calculate_hma(df_1d['close'].values, 21)
    
    # Align HTF to LTF (Rule 2 - auto shift(1) for completed bars)
    hma_4h_21_aligned = align_htf_to_ltf(prices, df_4h, hma_4h_21)
    hma_1d_21_aligned = align_htf_to_ltf(prices, df_1d, hma_1d_21)
    
    # Calculate 1h indicators
    atr_14 = calculate_atr(high, low, close, 14)
    rsi_14 = calculate_rsi(close, 14)
    vol_sma_20 = calculate_sma(volume, 20)
    
    signals = np.zeros(n)
    
    # Position sizing (Rule 4 - discrete levels, max 0.40)
    BASE_SIZE = 0.25
    
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
        
        if np.isnan(hma_4h_21_aligned[i]) or np.isnan(hma_1d_21_aligned[i]):
            continue
        
        if np.isnan(rsi_14[i]) or np.isnan(vol_sma_20[i]):
            continue
        
        # === 4H TREND DIRECTION (Primary signal) ===
        trend_4h_bullish = close[i] > hma_4h_21_aligned[i]
        trend_4h_bearish = close[i] < hma_4h_21_aligned[i]
        
        # === 1D MACRO BIAS (Confirmation) ===
        bias_1d_bullish = close[i] > hma_1d_21_aligned[i]
        bias_1d_bearish = close[i] < hma_1d_21_aligned[i]
        
        # === RSI PULLBACK (Entry timing) ===
        # Long: RSI pulled back to 35-55 within bullish trend
        rsi_pullback_long = 35 <= rsi_14[i] <= 55
        # Short: RSI pulled back to 45-65 within bearish trend
        rsi_pullback_short = 45 <= rsi_14[i] <= 65
        
        # === VOLUME CONFIRMATION ===
        vol_ok = volume[i] > 0.8 * vol_sma_20[i] if vol_sma_20[i] > 0 else False
        
        # === 4H TREND MOMENTUM (avoid entering when trend weakening) ===
        hma_4h_slope_long = False
        hma_4h_slope_short = False
        if i > 4:
            hma_4h_slope_long = hma_4h_21_aligned[i] > hma_4h_21_aligned[i-4]
            hma_4h_slope_short = hma_4h_21_aligned[i] < hma_4h_21_aligned[i-4]
        
        # === ENTRY LOGIC ===
        new_signal = 0.0
        bars_since_last_trade = i - last_trade_bar
        
        # LONG ENTRY: 4h bullish + 1d bullish bias + RSI pullback + volume
        # Core: 4h trend + RSI pullback (2 required)
        # Confirmation: 1d bias + volume + slope (need 1 of 3)
        long_core = trend_4h_bullish and rsi_pullback_long
        long_confirm = 0
        if bias_1d_bullish:
            long_confirm += 1
        if vol_ok:
            long_confirm += 1
        if hma_4h_slope_long:
            long_confirm += 1
        
        if long_core and long_confirm >= 1:
            new_signal = BASE_SIZE
        
        # SHORT ENTRY: 4h bearish + 1d bearish bias + RSI pullback + volume
        short_core = trend_4h_bearish and rsi_pullback_short
        short_confirm = 0
        if bias_1d_bearish:
            short_confirm += 1
        if vol_ok:
            short_confirm += 1
        if hma_4h_slope_short:
            short_confirm += 1
        
        if short_core and short_confirm >= 1:
            new_signal = -BASE_SIZE
        
        # === TRADE FREQUENCY SAFEGUARD ===
        # If no trades for 80 bars (~3-4 days on 1h), allow weaker entry
        if bars_since_last_trade > 80 and new_signal == 0.0 and not in_position:
            # Allow entry with just 4h trend + RSI (no 1d/volume required)
            if trend_4h_bullish and 30 <= rsi_14[i] <= 60:
                new_signal = BASE_SIZE * 0.6  # Smaller size
            elif trend_4h_bearish and 40 <= rsi_14[i] <= 70:
                new_signal = -BASE_SIZE * 0.6
        
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
                last_trade_bar = i
            elif np.sign(new_signal) != position_side:
                # Position flip
                position_side = np.sign(new_signal)
                entry_price = close[i]
                highest_price = close[i] if position_side > 0 else 0.0
                lowest_price = close[i] if position_side < 0 else 0.0
                last_trade_bar = i
        else:
            if in_position:
                # Exit position
                in_position = False
                position_side = 0
                entry_price = 0.0
                highest_price = 0.0
                lowest_price = 0.0
                last_trade_bar = i
        
        signals[i] = new_signal
    
    return signals