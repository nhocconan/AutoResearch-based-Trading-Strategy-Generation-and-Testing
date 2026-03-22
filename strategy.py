#!/usr/bin/env python3
"""
Experiment #013: 15m Multi-Timeframe Trend Pullback with Volume Confirmation

Hypothesis: After 12 failed experiments, the pattern shows lower TFs (15m/30m) fail
due to noise and fee drag. However, the BEST strategy so far (#002) was 30m with
4h HMA + RSI pullback. This suggests MTF filtering is critical for lower TFs.

This 15m strategy combines:

1. 4h HMA(21) trend bias: Stable HTF filter. Only long if price>4h_HMA, only short
   if price<4h_HMA. Reduces whipsaw from 15m noise.

2. 1h RSI(14) pullback: Enter on RSI pullback to 40-50 in uptrend, 50-60 in downtrend.
   More stable than 15m RSI, less lag than 4h RSI.

3. ADX(14) filter: Only trade when ADX>20 (some trend presence). Avoids chop.

4. Volume confirmation: Entry volume > 1.2 * 20-bar avg volume. Confirms breakout.

5. ATR(14) trailing stop: 2.5*ATR to protect from reversals.

6. Discrete position sizing: 0.25 base, 0.30 with strong confirmation.

Why this should beat #002 (Sharpe=0.123):
- 15m gives more precise entry timing than 30m
- 4h HMA + 1h RSI = proven MTF combination from #002
- Volume filter reduces false breakouts
- ADX filter avoids choppy periods that destroyed #001 and #007
- Target 40-70 trades/year on 15m (within fee drag limits)

Timeframe: 15m (REQUIRED for this experiment)
HTF: 4h and 1h via mtf_data helper (call ONCE before loop)
Position sizing: 0.25-0.30 discrete
Stoploss: 2.5 * ATR(14) trailing
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_15m_4h_hma_1h_rsi_adx_vol_atr_v1"
timeframe = "15m"
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
    """Calculate RSI using Wilder's smoothing."""
    close_s = pd.Series(close)
    delta = close_s.diff()
    gain = delta.where(delta > 0, 0)
    loss = -delta.where(delta < 0, 0)
    
    avg_gain = gain.ewm(span=period, min_periods=period, adjust=False).mean()
    avg_loss = loss.ewm(span=period, min_periods=period, adjust=False).mean()
    
    rs = avg_gain / avg_loss.replace(0, np.inf)
    rsi = 100 - (100 / (1 + rs))
    return rsi.values

def calculate_adx(high, low, close, period=14):
    """
    Calculate ADX (Average Directional Index).
    ADX > 25 = trending, ADX < 20 = ranging
    """
    high_s = pd.Series(high)
    low_s = pd.Series(low)
    close_s = pd.Series(close)
    
    # True Range
    tr1 = high_s - low_s
    tr2 = np.abs(high_s - close_s.shift(1))
    tr3 = np.abs(low_s - close_s.shift(1))
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    
    # Directional Movement
    plus_dm = high_s.diff()
    minus_dm = -low_s.diff()
    
    plus_dm = plus_dm.where((plus_dm > minus_dm) & (plus_dm > 0), 0)
    minus_dm = minus_dm.where((minus_dm > plus_dm) & (minus_dm > 0), 0)
    
    # Smoothed DM and TR
    atr = tr.ewm(span=period, min_periods=period, adjust=False).mean()
    plus_di = 100 * (plus_dm.ewm(span=period, min_periods=period, adjust=False).mean() / atr)
    minus_di = 100 * (minus_dm.ewm(span=period, min_periods=period, adjust=False).mean() / atr)
    
    # DX and ADX
    dx = 100 * np.abs(plus_di - minus_di) / (plus_di + minus_di).replace(0, np.inf)
    adx = dx.ewm(span=period, min_periods=period, adjust=False).mean()
    
    return adx.values

def calculate_volume_ma(volume, period=20):
    """Calculate rolling volume moving average."""
    vol_s = pd.Series(volume)
    vol_ma = vol_s.rolling(window=period, min_periods=period).mean().values
    return vol_ma

def generate_signals(prices):
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    volume = prices["volume"].values
    n = len(close)
    
    # Load HTF data ONCE before loop (Rule 1 - CRITICAL)
    # 4h for trend bias, 1h for RSI pullback timing
    df_4h = get_htf_data(prices, '4h')
    df_1h = get_htf_data(prices, '1h')
    
    # Calculate HTF indicators
    hma_4h = calculate_hma(df_4h['close'].values, 21)
    rsi_1h = calculate_rsi(df_1h['close'].values, 14)
    
    # Align HTF to LTF (Rule 2 - auto shift(1) for completed bars)
    hma_4h_aligned = align_htf_to_ltf(prices, df_4h, hma_4h)
    rsi_1h_aligned = align_htf_to_ltf(prices, df_1h, rsi_1h)
    
    # Calculate 15m indicators
    atr_14 = calculate_atr(high, low, close, 14)
    adx_14 = calculate_adx(high, low, close, 14)
    vol_ma_20 = calculate_volume_ma(volume, 20)
    
    signals = np.zeros(n)
    
    # Base position sizing (Rule 4 - discrete levels, max 0.40)
    BASE_SIZE = 0.25  # 25% of capital
    STRONG_SIZE = 0.30  # 30% with strong confirmation
    
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
        
        if np.isnan(hma_4h_aligned[i]):
            continue
        
        if np.isnan(rsi_1h_aligned[i]):
            continue
        
        if np.isnan(adx_14[i]):
            continue
        
        if np.isnan(vol_ma_20[i]) or vol_ma_20[i] == 0:
            continue
        
        # === 4H HMA TREND BIAS (HTF filter) ===
        bull_bias = close[i] > hma_4h_aligned[i]
        bear_bias = close[i] < hma_4h_aligned[i]
        
        # === 1H RSI PULLBACK SIGNALS ===
        # In uptrend: enter on RSI pullback to 40-50
        # In downtrend: enter on RSI rally to 50-60
        rsi_value = rsi_1h_aligned[i]
        rsi_pullback_long = 35 <= rsi_value <= 55
        rsi_pullback_short = 45 <= rsi_value <= 65
        
        # === ADX TREND STRENGTH FILTER ===
        # Only trade when ADX > 20 (some trend presence)
        adx_strong = adx_14[i] > 20
        
        # === VOLUME CONFIRMATION ===
        # Entry volume > 1.2 * 20-bar average
        vol_confirmation = volume[i] > 1.2 * vol_ma_20[i]
        
        # === ENTRY LOGIC ===
        new_signal = 0.0
        position_size = BASE_SIZE
        
        # LONG ENTRY: 4h bullish + 1h RSI pullback + ADX filter
        if bull_bias and rsi_pullback_long and adx_strong:
            # Strong confirmation with volume
            if vol_confirmation:
                position_size = STRONG_SIZE
            new_signal = position_size
        
        # SHORT ENTRY: 4h bearish + 1h RSI pullback + ADX filter
        elif bear_bias and rsi_pullback_short and adx_strong:
            # Strong confirmation with volume
            if vol_confirmation:
                position_size = STRONG_SIZE
            new_signal = -position_size
        
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
        trend_exit = False
        if in_position and position_side != 0:
            # Exit if HTF trend reverses against position
            if position_side > 0 and bear_bias:
                trend_exit = True
            if position_side < 0 and bull_bias:
                trend_exit = True
        
        # === ADX WEAKNESS EXIT ===
        adx_exit = False
        if in_position and position_side != 0:
            # Exit if ADX drops below 15 (trend dying)
            if adx_14[i] < 15:
                adx_exit = True
        
        # Apply stoploss or exit conditions
        if stoploss_triggered or trend_exit or adx_exit:
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