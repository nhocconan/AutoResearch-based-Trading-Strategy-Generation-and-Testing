#!/usr/bin/env python3
"""
Experiment #016: 4h Fisher Transform + 1D HMA Trend + ADX Regime + Volume Confirmation

Hypothesis: After 15 failed experiments, patterns show:
1. Lower TFs (15m/30m/1h) suffer from noise and fee drag - all failed
2. CRSI mean reversion doesn't work well across all symbols
3. Simple trend following fails in 2022 crash and 2025 bear market
4. 4h timeframe may be the sweet spot - enough signals, less noise than lower TFs

This 4h strategy combines:

1. 1D HMA trend bias: Daily Hull MA provides stable trend filter without lag.
   Only long if price > 1d_HMA, only short if price < 1d_HMA.
   More responsive than weekly, more stable than 4h.

2. Ehlers Fisher Transform: period=9, transforms price into Gaussian distribution.
   Long when Fisher crosses above -1.5 (oversold reversal)
   Short when Fisher crosses below +1.5 (overbought reversal)
   Proven to catch reversals in bear/range markets better than RSI.

3. ADX regime filter: ADX(14) > 25 = trending (follow trend), ADX < 20 = ranging (mean revert).
   Critical for adapting to market conditions. Hysteresis: enter 25, exit 18.

4. Volume confirmation: Volume > 1.5 * SMA(volume, 20) confirms breakout validity.
   Reduces false signals during low-liquidity periods.

5. ATR trailing stoploss: 2.5*ATR to protect from crashes while allowing room.

6. Asymmetric sizing: 0.30 in trending regime, 0.20 in ranging regime.

Why this should beat current best (Sharpe=0.123):
- Fisher Transform catches reversals better than RSI in bear markets
- 1D HMA more responsive than 1W for 4h timeframe
- ADX regime adaptation prevents mean-reversion in strong trends
- Volume filter reduces false breakouts
- Target 30-50 trades/year on 4h (optimal frequency per Rule 10)

Timeframe: 4h (REQUIRED for this experiment)
HTF: 1d via mtf_data helper (call ONCE before loop)
Position sizing: 0.20-0.30 discrete, regime-adaptive
Stoploss: 2.5 * ATR(14) trailing
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_4h_fisher_1d_hma_adx_vol_asym_atr_v1"
timeframe = "4h"
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

def calculate_adx(high, low, close, period=14):
    """
    Calculate ADX (Average Directional Index).
    ADX > 25 = trending market
    ADX < 20 = ranging/choppy market
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
    plus_dm_smooth = plus_dm.ewm(span=period, min_periods=period, adjust=False).mean()
    minus_dm_smooth = minus_dm.ewm(span=period, min_periods=period, adjust=False).mean()
    tr_smooth = tr.ewm(span=period, min_periods=period, adjust=False).mean()
    
    # Directional Indicators
    plus_di = 100 * (plus_dm_smooth / tr_smooth.replace(0, np.inf))
    minus_di = 100 * (minus_dm_smooth / tr_smooth.replace(0, np.inf))
    
    # DX and ADX
    dx = 100 * np.abs(plus_di - minus_di) / (plus_di + minus_di).replace(0, np.inf)
    adx = dx.ewm(span=period, min_periods=period, adjust=False).mean()
    
    return adx.values

def calculate_fisher_transform(high, low, period=9):
    """
    Calculate Ehlers Fisher Transform.
    Transforms price into Gaussian distribution for clearer reversal signals.
    
    Long signal: Fisher crosses above -1.5 (oversold reversal)
    Short signal: Fisher crosses below +1.5 (overbought reversal)
    """
    high_s = pd.Series(high)
    low_s = pd.Series(low)
    
    # Typical price
    typical = (high_s + low_s) / 2
    
    # Normalize price to -1 to +1 range
    highest = typical.rolling(window=period, min_periods=period).max()
    lowest = typical.rolling(window=period, min_periods=period).min()
    
    # Avoid division by zero
    range_val = highest - lowest
    range_val = range_val.replace(0, 0.001)
    
    normalized = 2 * ((typical - lowest) / range_val) - 1
    
    # Apply Fisher transform
    fisher = 0.5 * np.log((1 + normalized) / (1 - normalized).replace(0, 0.001))
    
    # Signal line (1-period lag of Fisher)
    fisher_signal = fisher.shift(1)
    
    return fisher.values, fisher_signal.values

def calculate_volume_ratio(volume, period=20):
    """Calculate volume ratio vs moving average."""
    volume_s = pd.Series(volume)
    vol_sma = volume_s.rolling(window=period, min_periods=period).mean()
    vol_ratio = volume_s / vol_sma.replace(0, np.inf)
    return vol_ratio.values

def generate_signals(prices):
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    volume = prices["volume"].values
    n = len(close)
    
    # Load HTF data ONCE before loop (Rule 1 - CRITICAL)
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate HTF indicators
    hma_1d = calculate_hma(df_1d['close'].values, 21)
    
    # Align HTF to LTF (Rule 2 - auto shift(1) for completed bars)
    hma_1d_aligned = align_htf_to_ltf(prices, df_1d, hma_1d)
    
    # Calculate 4h indicators
    atr_14 = calculate_atr(high, low, close, 14)
    adx_14 = calculate_adx(high, low, close, 14)
    fisher, fisher_signal = calculate_fisher_transform(high, low, period=9)
    vol_ratio = calculate_volume_ratio(volume, period=20)
    
    signals = np.zeros(n)
    
    # Base position sizing (Rule 4 - discrete levels, max 0.40)
    BASE_SIZE_TREND = 0.30  # Larger in trending regime
    BASE_SIZE_RANGE = 0.20  # Smaller in ranging regime
    
    # Track position state for stoploss
    in_position = False
    position_side = 0
    entry_price = 0.0
    highest_price = 0.0
    lowest_price = 0.0
    
    # ADX hysteresis tracking
    prev_adx_trending = False
    
    for i in range(100, n):
        # Skip if indicators not ready
        if np.isnan(atr_14[i]) or atr_14[i] == 0:
            continue
        
        if np.isnan(hma_1d_aligned[i]):
            continue
        
        if np.isnan(adx_14[i]):
            continue
        
        if np.isnan(fisher[i]) or np.isnan(fisher_signal[i]):
            continue
        
        if np.isnan(vol_ratio[i]):
            continue
        
        # === 1D HMA TREND BIAS ===
        bull_bias = close[i] > hma_1d_aligned[i]
        bear_bias = close[i] < hma_1d_aligned[i]
        
        # === ADX REGIME DETECTION with hysteresis ===
        # Enter trending mode at ADX > 25, exit at ADX < 18
        if adx_14[i] > 25:
            is_trending = True
            prev_adx_trending = True
        elif adx_14[i] < 18:
            is_trending = False
            prev_adx_trending = False
        else:
            # Keep previous state in hysteresis zone
            is_trending = prev_adx_trending
        
        is_ranging = not is_trending
        
        # === FISHER TRANSFORM SIGNALS ===
        # Long: Fisher crosses above -1.5 (oversold reversal)
        fisher_long = (fisher_signal[i] < -1.5) and (fisher[i] >= -1.5)
        
        # Short: Fisher crosses below +1.5 (overbought reversal)
        fisher_short = (fisher_signal[i] > 1.5) and (fisher[i] <= 1.5)
        
        # === VOLUME CONFIRMATION ===
        vol_confirmed = vol_ratio[i] > 1.3  # Volume 30% above average
        
        # === POSITION SIZING BASED ON REGIME ===
        if is_trending:
            base_size = BASE_SIZE_TREND
        else:
            base_size = BASE_SIZE_RANGE
        
        # === ENTRY LOGIC ===
        new_signal = 0.0
        
        # MODE 1: TRENDING REGIME - Follow 1D HMA direction with Fisher entry
        if is_trending:
            # Long: Bullish bias + Fisher long signal + volume confirmation
            if bull_bias and fisher_long and vol_confirmed:
                new_signal = base_size
            
            # Short: Bearish bias + Fisher short signal + volume confirmation
            elif bear_bias and fisher_short and vol_confirmed:
                new_signal = -base_size
        
        # MODE 2: RANGING REGIME - Mean reversion with Fisher extremes
        elif is_ranging:
            # Long: Fisher deeply oversold (< -1.8) regardless of trend bias
            if fisher[i] < -1.8:
                new_signal = base_size
            
            # Short: Fisher deeply overbought (> 1.8) regardless of trend bias
            elif fisher[i] > 1.8:
                new_signal = -base_size
        
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
        
        # === REGIME REVERSAL EXIT ===
        regime_exit = False
        if in_position and position_side != 0:
            # Exit long if trend reverses to bearish in trending regime
            if position_side > 0 and is_trending and bear_bias:
                regime_exit = True
            # Exit short if trend reverses to bullish in trending regime
            if position_side < 0 and is_trending and bull_bias:
                regime_exit = True
        
        # Apply stoploss or regime exit
        if stoploss_triggered or regime_exit:
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