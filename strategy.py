#!/usr/bin/env python3
"""
Experiment #007: 6h Volatility-Adaptive Regime Strategy

HYPOTHESIS: Markets cycle between high-vol trending and low-vol mean-reversion
regimes. ATR percentile (vs 20d history) directly measures this. In high-vol
regimes (>70th percentile), use trend-following with ATR expansion breakouts.
In low-vol regimes (<30th percentile), use mean-reversion with Bollinger Bands.
1d EMA200 filters direction. Volume confirms institutional participation.

WHY 6h: 4h strategies tend to overtrade. 12h/1d have too few opportunities.
6h provides balance: institutional-level signals, natural trade frequency.

WHY IT SHOULD WORK IN BULL + BEAR:
- Bull (2021): High-vol breakouts + EMA200 up = trend continuation longs
- Bear (2022): High-vol breakdowns + EMA200 down = trend continuation shorts
- Range (2023): Low-vol BB touches = mean-reversion fades
- Volatility crush after crashes: Low-vol regime catches reversals

Target: 60-100 total trades over 4 years (15-25/year).
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_6h_volatility_adaptive_regime_1d_v1"
timeframe = "6h"
leverage = 1.0

def calculate_atr(high, low, close, period=14):
    """Average True Range"""
    n = len(close)
    if n < period + 1:
        return np.full(n, np.nan)
    
    tr = np.zeros(n, dtype=np.float64)
    tr[0] = high[0] - low[0]
    for i in range(1, n):
        tr[i] = max(high[i] - low[i], abs(high[i] - close[i-1]), abs(low[i] - close[i-1]))
    
    atr = pd.Series(tr).ewm(span=period, min_periods=period, adjust=False).mean().values
    return atr

def calculate_atr_percentile(atr, period=20):
    """ATR percentile over rolling window - measures current volatility vs recent"""
    n = len(atr)
    if n < period:
        return np.full(n, np.nan)
    
    # Use ATR percentage of price for normalization
    percentile = np.full(n, np.nan)
    
    for i in range(period, n):
        window = atr[i-period:i] / 1000  # normalize
        current = atr[i] / 1000
        count_below = np.sum(window < current)
        percentile[i] = (count_below / period) * 100
    
    return percentile

def calculate_bollinger_bands(close, period=20, std_dev=2.0):
    """Bollinger Bands for mean-reversion signals"""
    mid = pd.Series(close).rolling(window=period, min_periods=period).mean().values
    std = pd.Series(close).rolling(window=period, min_periods=period).std().values
    upper = mid + std_dev * std
    lower = mid - std_dev * std
    return upper, mid, lower

def generate_signals(prices):
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    volume = prices["volume"].values
    n = len(close)
    
    # === Load HTF data ONCE before loop ===
    df_1d = get_htf_data(prices, '1d')
    
    # 1d EMA200 for multi-timeframe trend direction
    ema_200_1d = pd.Series(df_1d['close'].values).ewm(span=200, min_periods=200, adjust=False).mean().values
    ema_200_aligned = align_htf_to_ltf(prices, df_1d, ema_200_1d)
    
    # === Local 6h indicators ===
    atr_14 = calculate_atr(high, low, close, period=14)
    atr_percentile = calculate_atr_percentile(atr_14, period=20)
    bb_upper, bb_mid, bb_lower = calculate_bollinger_bands(close, period=20, std_dev=2.0)
    
    # Volume ratio (20-period MA for confirmation)
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_ratio = volume / np.where(vol_ma > 0, vol_ma, 1)
    
    # Signals
    signals = np.zeros(n)
    SIZE = 0.25
    
    # Position tracking
    in_position = False
    position_side = 0
    entry_price = 0.0
    entry_atr = 0.0
    entry_bar = 0
    
    # Regime tracking for hysteresis
    last_regime_bar = 0
    current_regime = 0  # 1=high_vol_trend, -1=low_vol_reversion, 0=neutral
    
    warmup = 200  # Need 200 for EMA200 alignment
    
    for i in range(warmup, n):
        # Skip if indicators not ready
        if np.isnan(atr_14[i]) or atr_14[i] <= 1e-10:
            signals[i] = 0.0
            in_position = False
            position_side = 0
            continue
        
        if np.isnan(ema_200_aligned[i]):
            signals[i] = 0.0
            in_position = False
            position_side = 0
            continue
        
        if np.isnan(atr_percentile[i]):
            signals[i] = 0.0
            in_position = False
            position_side = 0
            continue
        
        # === HTF TREND DIRECTION ===
        htf_bullish = close[i] > ema_200_aligned[i]
        htf_bearish = close[i] < ema_200_aligned[i]
        
        # === VOLATILITY REGIME (with hysteresis to avoid flipping) ===
        atr_pct = atr_percentile[i]
        
        # Only change regime every 5 bars (hysteresis)
        if i - last_regime_bar >= 5:
            if atr_pct > 70:
                current_regime = 1  # High volatility = trend following
            elif atr_pct < 30:
                current_regime = -1  # Low volatility = mean reversion
            # else: keep current regime
        
        # === VOLUME CONFIRMATION ===
        vol_spike = vol_ratio[i] > 1.5
        
        # === PRICE STRUCTURE ===
        # Donchian for trend signals
        donchian_20_up = pd.Series(high).rolling(window=20, min_periods=20).max().shift(1).values[i]
        donchian_20_lo = pd.Series(low).rolling(window=20, min_periods=20).min().shift(1).values[i]
        
        donchian_broken_up = close[i] > donchian_20_up if not np.isnan(donchian_20_up) else False
        donchian_broken_down = close[i] < donchian_20_lo if not np.isnan(donchian_20_lo) else False
        
        # BB position for mean-reversion signals
        bb_pos = (close[i] - bb_lower[i]) / (bb_upper[i] - bb_lower[i]) if not np.isnan(bb_upper[i] - bb_lower[i]) else 0.5
        
        # === ENTRY LOGIC ===
        desired_signal = 0.0
        
        if not in_position:
            # === HIGH VOLATILITY REGIME: TREND FOLLOWING ===
            if current_regime == 1:
                # Long: HTF uptrend + Donchian breakout + volume
                if htf_bullish and donchian_broken_up and vol_spike:
                    desired_signal = SIZE
                
                # Short: HTF downtrend + Donchian breakdown + volume
                if htf_bearish and donchian_broken_down and vol_spike:
                    desired_signal = -SIZE
            
            # === LOW VOLATILITY REGIME: MEAN REVERSION ===
            elif current_regime == -1:
                # Long: price at lower BB + recovering + volume
                if htf_bullish and bb_pos < 0.15 and vol_spike:
                    desired_signal = SIZE
                
                # Short: price at upper BB + declining + volume
                if htf_bearish and bb_pos > 0.85 and vol_spike:
                    desired_signal = -SIZE
        
        # === STOPLOSS (2.0 ATR) ===
        if in_position and position_side != 0:
            bars_held = i - entry_bar
            
            if position_side > 0:
                # Long stop: entry price - 2 ATR
                stop_price = entry_price - 2.0 * entry_atr
                if low[i] < stop_price:
                    desired_signal = 0.0
                # Exit if BB reaches upper band (mean reversion complete)
                elif bb_pos > 0.90 and current_regime == -1:
                    desired_signal = 0.0
                # Exit on regime change against us
                elif current_regime == -1 and htf_bearish:
                    desired_signal = 0.0
            
            elif position_side < 0:
                # Short stop: entry price + 2 ATR
                stop_price = entry_price + 2.0 * entry_atr
                if high[i] > stop_price:
                    desired_signal = 0.0
                # Exit if BB reaches lower band
                elif bb_pos < 0.10 and current_regime == -1:
                    desired_signal = 0.0
                # Exit on regime change against us
                elif current_regime == -1 and htf_bullish:
                    desired_signal = 0.0
        
        # === MINIMUM HOLD: 3 bars to reduce fee churn ===
        if in_position and (i - entry_bar) < 3:
            desired_signal = position_side * SIZE
        
        # === UPDATE POSITION ===
        if desired_signal != 0.0:
            if not in_position or np.sign(desired_signal) != position_side:
                # New position or flip
                in_position = True
                position_side = int(np.sign(desired_signal))
                entry_price = close[i]
                entry_atr = atr_14[i]
                entry_bar = i
                last_regime_bar = i
        else:
            if in_position:
                in_position = False
                position_side = 0
        
        signals[i] = desired_signal
    
    return signals