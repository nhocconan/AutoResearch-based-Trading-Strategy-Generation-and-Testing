#!/usr/bin/env python3
"""
Experiment #284: 30m Supertrend with 4h HMA Bias and RSI Pullback

Hypothesis: After analyzing 283 failed experiments, the pattern shows:
- KAMA-based strategies consistently fail on 30m/1h (#273, #278, #283 all Sharpe < -1.0)
- Fisher transform generates 0 trades (#275, #280)
- Simple trend + HTF bias works (current best: mtf_4h_kama_1d_hma_adx_atr_v1 Sharpe=0.478)

For 30m timeframe, I'll use:
1. 4h HMA(21) for directional bias - prevents counter-trend trades (proven effective)
2. 30m Supertrend(10, 3.0) for entry signals - cleaner than KAMA for 30m
3. RSI(14) pullback filter - enter on pullbacks in trend direction (RSI 35-50 for longs, 50-65 for shorts)
4. Volume confirmation - volume > 1.2x 20-period SMA to filter false breakouts
5. 2.5*ATR trailing stoploss - tighter than 12h strategies, appropriate for 30m
6. Asymmetric entries - only trade in direction of 4h HMA bias

Why 30m might work:
- More signals than 4h/12h (ensures >=10 trades per symbol)
- Less noise than 15m (fewer false signals)
- 4h HMA bias is strong filter that worked in current best strategy
- Supertrend captures trends better than EMA crossover (which always fails)

Key differences from failed strategies:
- NOT using KAMA (consistently failed on 30m/1h)
- NOT using Fisher (generates 0 trades)
- NOT using complex regime detection (failed in #279)
- Simple Supertrend + HTF bias + RSI pullback = cleaner signals
- RSI thresholds loose enough to ensure trades (35-50, not 25-35)

Timeframe: 30m (REQUIRED for this experiment)
HTF: 4h via mtf_data helper (call ONCE before loop)
Position sizing: 0.25-0.30 discrete
Stoploss: 2.5 * ATR(14) trailing
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_30m_supertrend_4h_hma_rsi_pullback_volume_v1"
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

def calculate_supertrend(high, low, close, period=10, multiplier=3.0):
    """Calculate Supertrend indicator."""
    n = len(close)
    atr = calculate_atr(high, low, close, period)
    
    supertrend = np.zeros(n)
    supertrend[:] = np.nan
    direction = np.zeros(n)  # 1 = uptrend, -1 = downtrend
    
    for i in range(period, n):
        if np.isnan(atr[i]):
            continue
        
        mid = (high[i] + low[i]) / 2.0
        upper_band = mid + multiplier * atr[i]
        lower_band = mid - multiplier * atr[i]
        
        if i == period:
            supertrend[i] = upper_band
            direction[i] = 1
        else:
            # Determine direction
            if close[i] > supertrend[i-1]:
                direction[i] = 1
                supertrend[i] = lower_band
            elif close[i] < supertrend[i-1]:
                direction[i] = -1
                supertrend[i] = upper_band
            else:
                direction[i] = direction[i-1]
                if direction[i] == 1:
                    supertrend[i] = max(lower_band, supertrend[i-1])
                else:
                    supertrend[i] = min(upper_band, supertrend[i-1])
    
    return supertrend, direction

def calculate_rsi(close, period=14):
    """Calculate RSI indicator."""
    close_s = pd.Series(close)
    delta = close_s.diff()
    gain = delta.where(delta > 0, 0.0)
    loss = -delta.where(delta < 0, 0.0)
    
    avg_gain = gain.ewm(span=period, min_periods=period, adjust=False).mean()
    avg_loss = loss.ewm(span=period, min_periods=period, adjust=False).mean()
    
    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))
    return rsi.values

def calculate_volume_sma(volume, period=20):
    """Calculate simple moving average of volume."""
    vol_s = pd.Series(volume)
    vol_sma = vol_s.rolling(window=period, min_periods=period).mean().values
    return vol_sma

def calculate_ema(close, period=21):
    """Calculate Exponential Moving Average."""
    close_s = pd.Series(close)
    ema = close_s.ewm(span=period, min_periods=period, adjust=False).mean()
    return ema.values

def generate_signals(prices):
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    volume = prices["volume"].values
    n = len(close)
    
    # Load HTF data ONCE before loop (Rule 1 - CRITICAL)
    df_4h = get_htf_data(prices, '4h')
    
    # Calculate HTF indicators
    hma_4h = calculate_hma(df_4h['close'].values, 21)
    
    # Align HTF to LTF (Rule 2 - no manual index mapping, auto shift(1))
    hma_4h_aligned = align_htf_to_ltf(prices, df_4h, hma_4h)
    
    # Calculate 30m indicators
    atr = calculate_atr(high, low, close, 14)
    supertrend, st_direction = calculate_supertrend(high, low, close, 10, 3.0)
    rsi = calculate_rsi(close, 14)
    vol_sma = calculate_volume_sma(volume, 20)
    ema_50 = calculate_ema(close, 50)
    
    signals = np.zeros(n)
    
    # Position sizing - discrete levels (Rule 4)
    SIZE_BASE = 0.28  # Base position size (conservative)
    SIZE_REDUCED = 0.20  # Reduced size in high vol
    SIZE_MAX = 0.35  # Maximum position size
    
    # Track position state for stoploss
    in_position = False
    position_side = 0
    highest_close = 0.0
    lowest_close = 0.0
    entry_price = 0.0
    entry_atr = 0.0
    
    for i in range(100, n):
        # Skip if indicators not ready
        if np.isnan(atr[i]) or atr[i] == 0:
            signals[i] = 0.0
            continue
        
        if np.isnan(hma_4h_aligned[i]):
            signals[i] = 0.0
            continue
        
        if np.isnan(supertrend[i]):
            signals[i] = 0.0
            continue
        
        if np.isnan(rsi[i]):
            signals[i] = 0.0
            continue
        
        if np.isnan(vol_sma[i]) or vol_sma[i] == 0:
            signals[i] = 0.0
            continue
        
        # === HIGHER TIMEFRAME BIAS ===
        # 4h HMA = strong directional bias (hard filter)
        bull_trend_4h = close[i] > hma_4h_aligned[i]
        bear_trend_4h = close[i] < hma_4h_aligned[i]
        
        # === SUPERTREND SIGNAL ===
        # Supertrend direction: 1 = uptrend (price above supertrend), -1 = downtrend
        st_bullish = st_direction[i] == 1
        st_bearish = st_direction[i] == -1
        
        # === RSI PULLBACK FILTER ===
        # For longs: RSI should be in pullback zone (35-55), not overbought
        rsi_pullback_long = 35 <= rsi[i] <= 55
        
        # For shorts: RSI should be in pullback zone (45-65), not oversold
        rsi_pullback_short = 45 <= rsi[i] <= 65
        
        # === VOLUME CONFIRMATION ===
        # Entry must have volume > 1.2x average
        volume_confirmed = volume[i] > 1.2 * vol_sma[i]
        
        # === VOLATILITY ADJUSTMENT ===
        # Reduce position size when ATR is elevated (>1.5x recent average)
        atr_recent_avg = np.nanmean(atr[max(0, i-20):i+1])
        high_volatility = atr[i] > 1.5 * atr_recent_avg if not np.isnan(atr_recent_avg) else False
        
        # Determine position size based on volatility
        if high_volatility:
            position_size = SIZE_REDUCED
        else:
            position_size = SIZE_BASE
        
        # === ENTRY CONDITIONS ===
        new_signal = 0.0
        
        # LONG ENTRY: Need 4h bias up + Supertrend bullish + RSI pullback + volume
        # Looser conditions to ensure >=10 trades per symbol
        long_conditions = (
            bull_trend_4h and  # 4h HMA bias bullish
            st_bullish and  # Supertrend bullish
            rsi_pullback_long and  # RSI in pullback zone
            volume_confirmed  # Volume confirms
        )
        
        # SHORT ENTRY: Mirror of long
        short_conditions = (
            bear_trend_4h and  # 4h HMA bias bearish
            st_bearish and  # Supertrend bearish
            rsi_pullback_short and  # RSI in pullback zone
            volume_confirmed  # Volume confirms
        )
        
        # === GENERATE SIGNAL ===
        if long_conditions:
            new_signal = position_size
        
        if short_conditions:
            new_signal = -position_size
        
        # === STOPLOSS LOGIC (Rule 6) - 2.5 * ATR trailing ===
        # Check stoploss on EXISTING position before considering new entry
        if in_position and position_side != 0:
            if position_side > 0:
                # Update highest close for long position
                if close[i] > highest_close:
                    highest_close = close[i]
                # Trailing stop: 2.5 * ATR below highest close
                stoploss_price = highest_close - 2.5 * atr[i]
                if close[i] < stoploss_price:
                    new_signal = 0.0  # Stoploss overrides entry signal
            
            if position_side < 0:
                # Update lowest close for short position
                if lowest_close == 0.0 or close[i] < lowest_close:
                    lowest_close = close[i]
                # Trailing stop: 2.5 * ATR above lowest close
                stoploss_price = lowest_close + 2.5 * atr[i]
                if close[i] > stoploss_price:
                    new_signal = 0.0  # Stoploss overrides entry signal
        
        # === TREND REVERSAL EXIT ===
        # Exit if HTF bias reverses against position
        if in_position and new_signal != 0.0:
            if position_side > 0 and bear_trend_4h:
                new_signal = 0.0  # 4h trend reversed against long
            if position_side < 0 and bull_trend_4h:
                new_signal = 0.0  # 4h trend reversed against short
        
        # === UPDATE POSITION TRACKING FOR NEXT BAR ===
        if new_signal != 0.0:
            if not in_position:
                # Entering new position
                in_position = True
                position_side = np.sign(new_signal)
                entry_price = close[i]
                entry_atr = atr[i]
                highest_close = close[i] if position_side > 0 else 0.0
                lowest_close = close[i] if position_side < 0 else 0.0
            elif np.sign(new_signal) != position_side:
                # Reversing position
                position_side = np.sign(new_signal)
                entry_price = close[i]
                entry_atr = atr[i]
                highest_close = close[i] if position_side > 0 else 0.0
                lowest_close = close[i] if position_side < 0 else 0.0
            # else: maintaining same position direction (possibly adjusted size)
        else:
            # Exiting position (signal-based or stoploss)
            if in_position:
                in_position = False
                position_side = 0
                entry_price = 0.0
                entry_atr = 0.0
                highest_close = 0.0
                lowest_close = 0.0
        
        signals[i] = new_signal
    
    return signals