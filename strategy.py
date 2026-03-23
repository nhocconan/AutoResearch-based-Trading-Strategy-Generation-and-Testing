#!/usr/bin/env python3
"""
Experiment #459: 4h Primary + 1d HTF — Fisher Transform + Vol Spike Reversion + Asymmetric Regime

Hypothesis: Based on research showing Ehlers Fisher Transform catches reversals in bear rallies
(Sharpe 0.8-1.5 through 2022 crash) and Vol Spike Reversion captures "vol crush" after panic.
Key innovations:
1. Ehlers Fisher Transform (period=9) for reversal entries — long when Fisher crosses above -1.5
2. Vol Spike Reversion: ATR(7)/ATR(30) > 2.0 + price < BB(20,2.5) → long (panic capitulation)
3. Asymmetric Regime: ADX>25 + price<SMA50 = bear (only short retrace to EMA21)
4. ADX<20 = range (mean revert at BB bounds) with hysteresis (enter 25, exit 18)
5. 1d HMA(21) for ultra-long-term bias filter
6. Position size: 0.25 base, 0.30 on strong confluence, discrete levels (0.0, ±0.25, ±0.30)
7. ATR(14) trailing stop at 2.5x for risk management

Target: Sharpe > 0.612, 20-50 trades/year on 4h, DD < -35%
Timeframe: 4h (proven best balance of signal quality vs fee drag)
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_4h_fisher_volspike_asymmetric_1d_v1"
timeframe = "4h"
leverage = 1.0

def calculate_hma(close, period):
    """Calculate Hull Moving Average."""
    close_s = pd.Series(close)
    half = period // 2
    wma1 = close_s.ewm(span=half, min_periods=half, adjust=False).mean()
    wma2 = close_s.ewm(span=period, min_periods=period, adjust=False).mean()
    diff = 2.0 * wma1 - wma2
    sqrt_period = int(np.sqrt(period))
    hma = diff.ewm(span=sqrt_period, min_periods=sqrt_period, adjust=False).mean()
    return hma.values

def calculate_ema(close, period):
    """Calculate EMA."""
    close_s = pd.Series(close)
    ema = close_s.ewm(span=period, min_periods=period, adjust=False).mean()
    return ema.values

def calculate_sma(close, period):
    """Calculate Simple Moving Average."""
    close_s = pd.Series(close)
    sma = close_s.rolling(window=period, min_periods=period).mean()
    return sma.values

def calculate_atr(high, low, close, period=14):
    """Calculate ATR using Wilder's smoothing."""
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]
    atr = pd.Series(tr).ewm(span=period, min_periods=period, adjust=False).mean().values
    return atr

def calculate_adx(high, low, close, period=14):
    """Calculate ADX (Average Directional Index)."""
    n = len(close)
    adx = np.full(n, np.nan)
    
    # Calculate True Range
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]
    
    # Calculate +DM and -DM
    plus_dm = np.zeros(n)
    minus_dm = np.zeros(n)
    
    for i in range(1, n):
        high_diff = high[i] - high[i-1]
        low_diff = low[i-1] - low[i]
        
        if high_diff > low_diff and high_diff > 0:
            plus_dm[i] = high_diff
        if low_diff > high_diff and low_diff > 0:
            minus_dm[i] = low_diff
    
    # Smooth with Wilder's method
    plus_dm_s = pd.Series(plus_dm).ewm(span=period, min_periods=period, adjust=False).mean().values
    minus_dm_s = pd.Series(minus_dm).ewm(span=period, min_periods=period, adjust=False).mean().values
    tr_s = pd.Series(tr).ewm(span=period, min_periods=period, adjust=False).mean().values
    
    # Calculate +DI and -DI
    with np.errstate(divide='ignore', invalid='ignore'):
        plus_di = 100.0 * plus_dm_s / (tr_s + 1e-10)
        minus_di = 100.0 * minus_dm_s / (tr_s + 1e-10)
    
    # Calculate DX
    with np.errstate(divide='ignore', invalid='ignore'):
        dx = 100.0 * np.abs(plus_di - minus_di) / (plus_di + minus_di + 1e-10)
    
    # Calculate ADX (smoothed DX)
    adx_s = pd.Series(dx).ewm(span=period, min_periods=period, adjust=False).mean()
    adx = adx_s.values
    
    return adx

def calculate_fisher_transform(high, low, period=9):
    """
    Calculate Ehlers Fisher Transform.
    Fisher = 0.5 * ln((1 + X) / (1 - X)) where X is normalized price
    Long when Fisher crosses above -1.5, short when crosses below +1.5
    """
    n = len(close := (high + low) / 2)
    fisher = np.full(n, np.nan)
    fisher_prev = np.full(n, np.nan)
    
    # Normalize price to -1 to +1 range
    for i in range(period, n):
        highest = np.nanmax(high[i-period+1:i+1])
        lowest = np.nanmin(low[i-period+1:i+1])
        price_range = highest - lowest
        
        if price_range > 1e-10:
            # Normalize to 0-1, then to -0.99 to +0.99
            x = 2.0 * ((close[i] - lowest) / price_range) - 1.0
            x = np.clip(x, -0.99, 0.99)
            
            # Fisher transform
            fisher[i] = 0.5 * np.log((1.0 + x) / (1.0 - x + 1e-10))
            fisher_prev[i] = fisher[i-1] if i > 0 else 0.0
        else:
            fisher[i] = fisher[i-1] if i > 0 else 0.0
            fisher_prev[i] = fisher_prev[i-1] if i > 0 else 0.0
    
    return fisher, fisher_prev

def calculate_bollinger_bands(close, period=20, std_mult=2.5):
    """Calculate Bollinger Bands."""
    close_s = pd.Series(close)
    sma = close_s.rolling(window=period, min_periods=period).mean()
    std = close_s.rolling(window=period, min_periods=period).std()
    
    upper = sma + std_mult * std
    lower = sma - std_mult * std
    
    return upper.values, lower.values, sma.values

def calculate_vol_spike_ratio(atr, short_period=7, long_period=30):
    """Calculate ATR ratio for vol spike detection."""
    n = len(atr)
    ratio = np.full(n, np.nan)
    
    atr_s = pd.Series(atr)
    atr_short = atr_s.ewm(span=short_period, min_periods=short_period, adjust=False).mean().values
    atr_long = atr_s.ewm(span=long_period, min_periods=long_period, adjust=False).mean().values
    
    with np.errstate(divide='ignore', invalid='ignore'):
        ratio = atr_short / (atr_long + 1e-10)
    
    return ratio

def generate_signals(prices):
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    n = len(close)
    
    # Load HTF data ONCE before loop (Rule 1 - CRITICAL)
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate 4h indicators (primary timeframe)
    atr_14 = calculate_atr(high, low, close, period=14)
    adx_14 = calculate_adx(high, low, close, period=14)
    fisher, fisher_prev = calculate_fisher_transform(high, low, period=9)
    bb_upper, bb_lower, bb_mid = calculate_bollinger_bands(close, period=20, std_mult=2.5)
    vol_spike_ratio = calculate_vol_spike_ratio(atr_14, short_period=7, long_period=30)
    
    ema_21 = calculate_ema(close, 21)
    sma_50 = calculate_sma(close, 50)
    sma_200 = calculate_sma(close, 200)
    
    # Calculate and align HTF HMA for bias (1d)
    hma_1d_raw = calculate_hma(df_1d['close'].values, 21)
    hma_1d_aligned = align_htf_to_ltf(prices, df_1d, hma_1d_raw)
    
    # Calculate median ATR for vol normalization
    valid_atr = atr_14[100:]
    atr_median = np.nanmedian(valid_atr[~np.isnan(valid_atr)])
    if np.isnan(atr_median) or atr_median <= 0:
        atr_median = np.nanmean(valid_atr[~np.isnan(valid_atr)])
    
    signals = np.zeros(n)
    BASE_SIZE = 0.25  # 25% base position size
    
    # Position tracking for stoploss
    in_position = False
    position_side = 0
    entry_price = 0.0
    entry_atr = 0.0
    highest_since_entry = 0.0
    lowest_since_entry = float('inf')
    
    # Regime hysteresis tracking
    prev_adx_regime = 0  # 0=unknown, 1=trending, 2=ranging
    
    for i in range(300, n):
        # Skip if indicators not ready
        if np.isnan(atr_14[i]) or atr_14[i] <= 1e-10:
            continue
        if np.isnan(adx_14[i]) or np.isnan(fisher[i]):
            continue
        if np.isnan(bb_upper[i]) or np.isnan(bb_lower[i]):
            continue
        if np.isnan(hma_1d_aligned[i]) or np.isnan(sma_50[i]):
            continue
        
        # === REGIME DETECTION with HYSTERESIS (ADX) ===
        # Enter trending at ADX>25, exit at ADX<18
        if prev_adx_regime == 1:  # Was trending
            if adx_14[i] < 18:
                prev_adx_regime = 2  # Switch to ranging
            else:
                prev_adx_regime = 1
        elif prev_adx_regime == 2:  # Was ranging
            if adx_14[i] > 25:
                prev_adx_regime = 1  # Switch to trending
            else:
                prev_adx_regime = 2
        else:  # Unknown
            if adx_14[i] > 25:
                prev_adx_regime = 1
            elif adx_14[i] < 20:
                prev_adx_regime = 2
        
        regime_trending = (prev_adx_regime == 1)
        regime_ranging = (prev_adx_regime == 2)
        
        # === HTF TREND BIAS (1d HMA) ===
        price_above_hma_1d = close[i] > hma_1d_aligned[i]
        price_below_hma_1d = close[i] < hma_1d_aligned[i]
        
        # === BEAR REGIME DETECTION (Asymmetric Logic) ===
        bear_regime = price_below_hma_1d and close[i] < sma_50[i]
        bull_regime = price_above_hma_1d and close[i] > sma_50[i]
        
        # === VOL SPIKE DETECTION ===
        vol_spike = vol_spike_ratio[i] > 2.0
        
        # === FISHER TRANSFORM SIGNALS ===
        fisher_cross_up = (fisher_prev[i] < -1.5) and (fisher[i] >= -1.5)
        fisher_cross_down = (fisher_prev[i] > 1.5) and (fisher[i] <= 1.5)
        fisher_oversold = fisher[i] < -1.5
        fisher_overbought = fisher[i] > 1.5
        
        # === BOLLINGER BAND SIGNALS ===
        price_below_bb_lower = close[i] < bb_lower[i]
        price_above_bb_upper = close[i] > bb_upper[i]
        price_at_bb_lower = close[i] < (bb_lower[i] * 1.005)  # Within 0.5%
        price_at_bb_upper = close[i] > (bb_upper[i] * 0.995)
        
        # === VOL FILTER ===
        vol_ratio = atr_14[i] / (atr_median + 1e-10)
        if vol_ratio > 2.5:
            position_size = BASE_SIZE * 0.5
        elif vol_ratio > 1.5:
            position_size = BASE_SIZE * 0.75
        else:
            position_size = BASE_SIZE
        
        # === DESIRED SIGNAL ===
        desired_signal = 0.0
        signal_strength = 0
        
        # === REGIME 1: RANGING (ADX < 20) — MEAN REVERSION ===
        if regime_ranging:
            # Long: Price at BB lower + Fisher oversold + HTF not strongly bearish
            if price_at_bb_lower and fisher_oversold:
                signal_strength = 1
                if not bear_regime:
                    signal_strength += 2
                if vol_spike:
                    signal_strength += 1  # Panic capitulation
                
                if signal_strength >= 2:
                    desired_signal = position_size * (0.8 + 0.2 * min(signal_strength, 4) / 4)
            
            # Short: Price at BB upper + Fisher overbought + HTF not strongly bullish
            if price_at_bb_upper and fisher_overbought and desired_signal == 0:
                signal_strength = 1
                if not bull_regime:
                    signal_strength += 2
                
                if signal_strength >= 2:
                    desired_signal = -position_size * (0.8 + 0.2 * min(signal_strength, 4) / 4)
        
        # === REGIME 2: TRENDING (ADX > 25) — ASYMMETRIC LOGIC ===
        elif regime_trending:
            # BEAR REGIME: Only short retraces to EMA21 (no longs)
            if bear_regime:
                # Short: Retrace to EMA21 + Fisher cross down
                if close[i] > ema_21[i] and close[i] < ema_21[i] * 1.01:
                    if fisher_cross_down or fisher_overbought:
                        signal_strength = 2
                        if adx_14[i] > 30:
                            signal_strength += 1
                        
                        desired_signal = -position_size * (0.8 + 0.2 * min(signal_strength, 3) / 3)
                
                # Vol spike reversion in bear (panic long only on extreme)
                if vol_spike and price_below_bb_lower and fisher_cross_up:
                    signal_strength = 3  # Strong signal
                    desired_signal = position_size * 0.6  # Smaller size for counter-trend
            
            # BULL REGIME: Only long retraces to EMA21 (no shorts)
            elif bull_regime:
                # Long: Retrace to EMA21 + Fisher cross up
                if close[i] < ema_21[i] * 1.01 and close[i] > ema_21[i] * 0.99:
                    if fisher_cross_up or fisher_oversold:
                        signal_strength = 2
                        if adx_14[i] > 30:
                            signal_strength += 1
                        
                        desired_signal = position_size * (0.8 + 0.2 * min(signal_strength, 3) / 3)
            
            # NEUTRAL: Use Fisher crosses with HTF bias
            else:
                if fisher_cross_up and price_above_hma_1d:
                    desired_signal = position_size * 0.7
                elif fisher_cross_down and price_below_hma_1d:
                    desired_signal = -position_size * 0.7
        
        # === VOL SPIKE REVERSION (All Regimes) — Panic Capitulation ===
        # This overrides other signals when vol spike is extreme
        if vol_spike and vol_spike_ratio[i] > 2.5:
            if price_below_bb_lower and fisher_oversold and close[i] < sma_200[i]:
                # Extreme panic in bear market — strong long signal
                desired_signal = max(desired_signal, position_size * 0.8)
        
        # === CAP SIGNAL TO MAX 0.35 ===
        if desired_signal > 0.35:
            desired_signal = 0.35
        elif desired_signal < -0.35:
            desired_signal = -0.35
        
        # === STOPLOSS CHECK (Trailing ATR) ===
        stoploss_triggered = False
        
        if in_position and position_side > 0:
            highest_since_entry = max(highest_since_entry, close[i])
            stop_price = highest_since_entry - 2.5 * entry_atr
            if close[i] < stop_price:
                stoploss_triggered = True
        
        if in_position and position_side < 0:
            lowest_since_entry = min(lowest_since_entry, close[i])
            stop_price = lowest_since_entry + 2.5 * entry_atr
            if close[i] > stop_price:
                stoploss_triggered = True
        
        if stoploss_triggered:
            desired_signal = 0.0
        
        # === FISHER EXTREME EXIT (Take Profit) ===
        if in_position and position_side > 0 and fisher[i] > 1.5:
            desired_signal = 0.0
        
        if in_position and position_side < 0 and fisher[i] < -1.5:
            desired_signal = 0.0
        
        # === HOLD LOGIC — Maintain position if bias unchanged ===
        if in_position and desired_signal == 0.0 and not stoploss_triggered:
            if position_side > 0 and (price_above_hma_1d or bull_regime):
                desired_signal = position_size
            elif position_side < 0 and (price_below_hma_1d or bear_regime):
                desired_signal = -position_size
        
        # === DISCRETIZE SIGNAL VALUES ===
        if desired_signal != 0.0:
            if desired_signal > 0:
                if desired_signal >= 0.28:
                    desired_signal = 0.30
                elif desired_signal >= 0.22:
                    desired_signal = 0.25
                else:
                    desired_signal = 0.15
            else:
                if desired_signal <= -0.28:
                    desired_signal = -0.30
                elif desired_signal <= -0.22:
                    desired_signal = -0.25
                else:
                    desired_signal = -0.15
        
        # === UPDATE POSITION TRACKING ===
        if desired_signal != 0.0:
            if not in_position:
                in_position = True
                position_side = int(np.sign(desired_signal))
                entry_price = close[i]
                entry_atr = atr_14[i]
                highest_since_entry = close[i] if position_side > 0 else 0.0
                lowest_since_entry = close[i] if position_side < 0 else float('inf')
            elif np.sign(desired_signal) != position_side:
                position_side = int(np.sign(desired_signal))
                entry_price = close[i]
                entry_atr = atr_14[i]
                highest_since_entry = close[i] if position_side > 0 else 0.0
                lowest_since_entry = close[i] if position_side < 0 else float('inf')
        else:
            if in_position:
                in_position = False
                position_side = 0
                entry_price = 0.0
                entry_atr = 0.0
                highest_since_entry = 0.0
                lowest_since_entry = float('inf')
        
        signals[i] = desired_signal
    
    return signals