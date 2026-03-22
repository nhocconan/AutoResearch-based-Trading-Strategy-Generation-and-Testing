#!/usr/bin/env python3
"""
Experiment #276: 1d Donchian Breakout with 1w HMA Regime Filter

Hypothesis: Daily timeframe offers cleaner signals with less noise than lower TFs.
After analyzing 275 experiments, the pattern shows:
- 1d primary strategies (#264, #270) achieved positive Sharpe (0.05-0.14)
- HTF bias is CRITICAL for avoiding counter-trend trades in 2022 crash
- Simple breakout + volume + regime filter works better than complex ensembles

This strategy uses:
1. 1d Donchian(20) breakout - captures sustained momentum moves
2. 1w HMA(21) for regime bias - strongest directional filter available
3. Volume confirmation (>1.2x 20-period avg) - validates breakout strength
4. 3.5*ATR(14) trailing stoploss - appropriate width for daily bars
5. Asymmetric entries - only trade in direction of 1w HMA regime
6. Discrete position sizing (0.25-0.35) - minimizes fee churn

Why 1d should work:
- Daily bars filter out intraday noise and fake breakouts
- Fewer signals = lower fee drag, higher quality trades
- 1w HMA provides strongest regime filter (weekly trend is hard to fake)
- Still generates enough trades (10+ per symbol over 4 years)

Key differences from failed strategies:
- NO RSI (pullback strategies consistently failed)
- NO complex voting/ensemble (ensembles underperformed simple trend)
- NO Choppiness/Fisher (added no value in prior experiments)
- Simple breakout + volume + 1w bias = cleaner, fewer but higher quality signals

Timeframe: 1d (REQUIRED for this experiment)
HTF: 1w via mtf_data helper (call ONCE before loop)
Position sizing: 0.25-0.35 discrete
Stoploss: 3.5 * ATR(14) trailing
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_1d_donchian_breakout_1w_hma_volume_atr_v1"
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
    """Calculate Donchian Channel (highest high, lowest low over period)."""
    n = len(high)
    upper = np.full(n, np.nan)
    lower = np.full(n, np.nan)
    
    for i in range(period - 1, n):
        upper[i] = np.max(high[i - period + 1:i + 1])
        lower[i] = np.min(low[i - period + 1:i + 1])
    
    return upper, lower

def calculate_volume_sma(volume, period=20):
    """Calculate simple moving average of volume."""
    vol_s = pd.Series(volume)
    vol_sma = vol_s.rolling(window=period, min_periods=period).mean().values
    return vol_sma

def calculate_ema(close, period=50):
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
    df_1w = get_htf_data(prices, '1w')
    
    # Calculate HTF indicators
    hma_1w = calculate_hma(df_1w['close'].values, 21)
    
    # Align HTF to LTF (Rule 2 - no manual index mapping, auto shift(1))
    hma_1w_aligned = align_htf_to_ltf(prices, df_1w, hma_1w)
    
    # Calculate 1d indicators
    atr = calculate_atr(high, low, close, 14)
    donchian_upper, donchian_lower = calculate_donchian(high, low, 20)
    vol_sma = calculate_volume_sma(volume, 20)
    ema_50 = calculate_ema(close, 50)
    
    signals = np.zeros(n)
    
    # Position sizing - discrete levels (Rule 4)
    SIZE_BASE = 0.30  # Base position size
    SIZE_REDUCED = 0.25  # Reduced size in high vol
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
        
        if np.isnan(hma_1w_aligned[i]):
            signals[i] = 0.0
            continue
        
        if np.isnan(donchian_upper[i]) or np.isnan(donchian_lower[i]):
            signals[i] = 0.0
            continue
        
        if np.isnan(vol_sma[i]) or vol_sma[i] == 0:
            signals[i] = 0.0
            continue
        
        # === HIGHER TIMEFRAME REGIME BIAS ===
        # 1w HMA = strongest directional bias (hard filter)
        bull_regime_1w = close[i] > hma_1w_aligned[i]
        bear_regime_1w = close[i] < hma_1w_aligned[i]
        
        # === VOLUME CONFIRMATION ===
        # Breakout must have volume > 1.2x average to be valid
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
        
        # === DONCHIAN BREAKOUT SIGNALS ===
        # Long breakout: price breaks above Donchian upper (previous bar)
        breakout_long = close[i] > donchian_upper[i-1] if not np.isnan(donchian_upper[i-1]) else False
        
        # Short breakout: price breaks below Donchian lower (previous bar)
        breakout_short = close[i] < donchian_lower[i-1] if not np.isnan(donchian_lower[i-1]) else False
        
        # === ENTRY CONDITIONS ===
        new_signal = 0.0
        
        # LONG ENTRY: Need 1w regime up + Donchian breakout + volume confirmation
        # Looser conditions to ensure >=10 trades per symbol on daily data
        long_conditions = (
            bull_regime_1w and  # 1w HMA regime bullish
            breakout_long and  # Donchian breakout
            volume_confirmed  # Volume confirms breakout
        )
        
        # SHORT ENTRY: Mirror of long
        short_conditions = (
            bear_regime_1w and  # 1w HMA regime bearish
            breakout_short and  # Donchian breakout
            volume_confirmed  # Volume confirms breakout
        )
        
        # === GENERATE SIGNAL ===
        if long_conditions:
            new_signal = position_size
        
        if short_conditions:
            new_signal = -position_size
        
        # === STOPLOSS LOGIC (Rule 6) - 3.5 * ATR trailing ===
        # Check stoploss on EXISTING position before considering new entry
        if in_position and position_side != 0:
            if position_side > 0:
                # Update highest close for long position
                if close[i] > highest_close:
                    highest_close = close[i]
                # Trailing stop: 3.5 * ATR below highest close
                stoploss_price = highest_close - 3.5 * atr[i]
                if close[i] < stoploss_price:
                    new_signal = 0.0  # Stoploss overrides entry signal
            
            if position_side < 0:
                # Update lowest close for short position
                if lowest_close == 0.0 or close[i] < lowest_close:
                    lowest_close = close[i]
                # Trailing stop: 3.5 * ATR above lowest close
                stoploss_price = lowest_close + 3.5 * atr[i]
                if close[i] > stoploss_price:
                    new_signal = 0.0  # Stoploss overrides entry signal
        
        # === REGIME REVERSAL EXIT ===
        # Exit if 1w regime reverses against position
        if in_position and new_signal != 0.0:
            if position_side > 0 and bear_regime_1w:
                new_signal = 0.0  # 1w regime reversed against long
            if position_side < 0 and bull_regime_1w:
                new_signal = 0.0  # 1w regime reversed against short
        
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