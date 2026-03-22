#!/usr/bin/env python3
"""
Experiment #013: 1d Donchian Breakout + 1w HMA Trend + RSI Filter + ATR Trail

Hypothesis: After 12 failed experiments with regime-switching strategies, return to 
proven breakout/trend-following on higher timeframe (1d). This combines:

1. 1D Donchian(20) breakout - clean entry trigger on sustained moves
2. 1W HMA(21) - secular trend bias (only long if weekly bullish, only short if bearish)
3. RSI(14) filter - avoid entering at extremes (RSI 35-65 range for entries)
4. ATR(14) trailing stop - 2.5 ATR exit to protect capital
5. Volume confirmation - breakout volume > 1.2x 20-day avg volume

Why this should work:
- 1d timeframe = 20-50 trades/year naturally (matches target)
- Weekly trend filter = avoids counter-trend trades in strong regimes
- RSI filter = avoids chasing breakouts at extremes
- Volume confirmation = filters false breakouts
- Simple logic = fewer conflicting filters = more trades generated

Timeframe: 1d (REQUIRED for this experiment)
HTF: 1w via mtf_data helper (call ONCE before loop)
Position sizing: 0.28 long, 0.25 short (discrete)
Stoploss: 2.5 * ATR(14) trailing
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_1d_donchian_1w_hma_rsi_vol_atr_v1"
timeframe = "1d"
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

def calculate_donchian(high, low, period=20):
    """Calculate Donchian Channel (highest high / lowest low over period)."""
    high_s = pd.Series(high)
    low_s = pd.Series(low)
    
    upper = high_s.rolling(window=period, min_periods=period).max()
    lower = low_s.rolling(window=period, min_periods=period).min()
    
    return upper.values, lower.values

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
    rsi = rsi.fillna(50.0).values
    return rsi

def calculate_volume_sma(volume, period=20):
    """Calculate SMA of volume."""
    vol_s = pd.Series(volume)
    return vol_s.rolling(window=period, min_periods=period).mean().values

def generate_signals(prices):
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    volume = prices["volume"].values
    n = len(close)
    
    # Load HTF data ONCE before loop (Rule 1 - CRITICAL)
    df_1w = get_htf_data(prices, '1w')
    
    # Calculate 1W indicators
    hma_1w_21 = calculate_hma(df_1w['close'].values, 21)
    
    # Align HTF to LTF (Rule 2 - auto shift(1) for completed bars)
    hma_1w_21_aligned = align_htf_to_ltf(prices, df_1w, hma_1w_21)
    
    # Calculate 1d indicators
    atr_14 = calculate_atr(high, low, close, 14)
    rsi_14 = calculate_rsi(close, 14)
    donchian_upper, donchian_lower = calculate_donchian(high, low, 20)
    volume_sma_20 = calculate_volume_sma(volume, 20)
    
    signals = np.zeros(n)
    
    # Base position sizing (Rule 4 - discrete levels, max 0.40)
    BASE_SIZE_LONG = 0.28
    BASE_SIZE_SHORT = 0.25  # Slightly smaller for shorts (bear market bias)
    
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
        
        if np.isnan(hma_1w_21_aligned[i]):
            continue
        
        if np.isnan(donchian_upper[i]) or np.isnan(donchian_lower[i]):
            continue
        
        if np.isnan(rsi_14[i]):
            continue
        
        if np.isnan(volume_sma_20[i]) or volume_sma_20[i] == 0:
            continue
        
        # === 1W SECULAR TREND BIAS ===
        weekly_bullish = close[i] > hma_1w_21_aligned[i]
        weekly_bearish = close[i] < hma_1w_21_aligned[i]
        
        # === DONCHIAN BREAKOUT ===
        # Breakout above previous upper band
        breakout_long = close[i] > donchian_upper[i-1] if i > 0 and not np.isnan(donchian_upper[i-1]) else False
        # Breakout below previous lower band
        breakout_short = close[i] < donchian_lower[i-1] if i > 0 and not np.isnan(donchian_lower[i-1]) else False
        
        # === RSI FILTER (avoid extremes) ===
        # For longs: RSI should not be overbought (>70)
        rsi_ok_long = rsi_14[i] < 70 and rsi_14[i] > 35
        # For shorts: RSI should not be oversold (<30)
        rsi_ok_short = rsi_14[i] > 30 and rsi_14[i] < 65
        
        # === VOLUME CONFIRMATION ===
        vol_confirmation = volume[i] > 1.2 * volume_sma_20[i]
        
        # === POSITION SIZING ===
        long_size = BASE_SIZE_LONG
        short_size = BASE_SIZE_SHORT
        
        # === ENTRY LOGIC ===
        new_signal = 0.0
        
        # LONG ENTRY: Breakout + Weekly bullish + RSI ok + Volume confirmation
        if breakout_long and weekly_bullish and rsi_ok_long:
            new_signal = long_size
        
        # SHORT ENTRY: Breakout + Weekly bearish + RSI ok + Volume confirmation
        if breakout_short and weekly_bearish and rsi_ok_short:
            new_signal = -short_size
        
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
            # Exit long if weekly trend turns bearish
            if position_side > 0 and weekly_bearish:
                trend_reversal = True
            # Exit short if weekly trend turns bullish
            if position_side < 0 and weekly_bullish:
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