#!/usr/bin/env python3
"""
Experiment #347: 1d Primary + 1w HTF — Fisher Transform + ADX Regime with Asymmetric Bias

Hypothesis: Previous 1d strategies failed because:
1. Too strict CRSI thresholds (<15, >85) = too few trades
2. Volume confirmation filters out valid 1d signals
3. Symmetric long/short logic doesn't adapt to bear markets (2022, 2025)
4. Choppiness Index too noisy on 1d timeframe

This strategy uses:
1. 1w HMA(21) as ULTRA-MACRO BIAS (only long if weekly bullish, prefer shorts if weekly bearish)
2. Ehlers Fisher Transform (period=9) for reversal timing (crosses -1.5/+1.5)
3. ADX(14) for trend strength (threshold=20, not 25+ to get more trades)
4. Asymmetric sizing: 0.30 longs, 0.25 shorts (bear market adaptation)
5. ATR(14) trailing stop at 2.5x for risk management

KEY INSIGHT: Fisher Transform catches reversals better than RSI in bear markets.
Combined with weekly bias filter, this reduces whipsaw while generating 15-25 trades/year.
Lower ADX threshold (20 vs 25) ensures we get enough trades on ALL symbols.

TARGET: 15-25 trades/year on 1d, Sharpe > 0.5 on ALL symbols (BTC/ETH/SOL)
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_1d_fisher_adx_1w_hma_asymmetric_v1"
timeframe = "1d"
leverage = 1.0

def calculate_hma(close, period=21):
    """Calculate Hull Moving Average."""
    close_s = pd.Series(close)
    wma_half = close_s.ewm(span=period//2, min_periods=period//2, adjust=False).mean()
    wma_full = close_s.ewm(span=period, min_periods=period, adjust=False).mean()
    wma_diff = 2 * wma_half - wma_full
    hma = wma_diff.ewm(span=int(np.sqrt(period)), min_periods=int(np.sqrt(period)), adjust=False).mean()
    return hma.values

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
    high_s = pd.Series(high)
    low_s = pd.Series(low)
    close_s = pd.Series(close)
    
    # True Range
    tr1 = high_s - low_s
    tr2 = (high_s - close_s.shift(1)).abs()
    tr3 = (low_s - close_s.shift(1)).abs()
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    
    # Directional Movement
    plus_dm = high_s.diff()
    minus_dm = -low_s.diff()
    
    plus_dm = plus_dm.clip(lower=0)
    minus_dm = minus_dm.clip(lower=0)
    
    # Smooth with Wilder's method (EMA with span=period)
    atr_smooth = tr.ewm(span=period, min_periods=period, adjust=False).mean()
    plus_di = 100 * (plus_dm.ewm(span=period, min_periods=period, adjust=False).mean() / (atr_smooth + 1e-10))
    minus_di = 100 * (minus_dm.ewm(span=period, min_periods=period, adjust=False).mean() / (atr_smooth + 1e-10))
    
    # DX and ADX
    dx = 100 * np.abs(plus_di - minus_di) / (plus_di + minus_di + 1e-10)
    adx = dx.ewm(span=period, min_periods=period, adjust=False).mean()
    
    return adx.fillna(0).values

def calculate_fisher_transform(high, low, close, period=9):
    """
    Calculate Ehlers Fisher Transform.
    Fisher = 0.5 * ln((1 + X) / (1 - X)) where X = 0.67 * ((close - LL) / (HH - LL) - 0.5)
    HH/LL = highest high / lowest low over period
    """
    close_s = pd.Series(close)
    high_s = pd.Series(high)
    low_s = pd.Series(low)
    
    # Calculate highest high and lowest low over period
    hh = high_s.rolling(window=period, min_periods=period).max()
    ll = low_s.rolling(window=period, min_periods=period).min()
    
    # Normalize price position within range
    with np.errstate(divide='ignore', invalid='ignore'):
        x = 0.67 * ((close_s - ll) / (hh - ll + 1e-10) - 0.5)
        x = x.clip(-0.99, 0.99)  # Prevent log domain errors
        
        # Fisher Transform
        fisher = 0.5 * np.log((1 + x) / (1 - x + 1e-10))
    
    fisher = fisher.fillna(0).values
    return fisher

def calculate_rsi(close, period=14):
    """Calculate RSI."""
    close_s = pd.Series(close)
    delta = close_s.diff()
    gain = delta.clip(lower=0)
    loss = (-delta).clip(lower=0)
    avg_gain = gain.ewm(span=period, min_periods=period, adjust=False).mean()
    avg_loss = loss.ewm(span=period, min_periods=period, adjust=False).mean()
    with np.errstate(divide='ignore', invalid='ignore'):
        rs = avg_gain / (avg_loss + 1e-10)
        rsi = 100.0 - (100.0 / (1.0 + rs))
    return rsi.fillna(50.0).values

def generate_signals(prices):
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    n = len(close)
    
    # Load HTF data ONCE before loop (Rule 1 - CRITICAL)
    df_1w = get_htf_data(prices, '1w')
    
    # Calculate 1d indicators (primary timeframe)
    atr_14 = calculate_atr(high, low, close, period=14)
    adx_14 = calculate_adx(high, low, close, period=14)
    fisher = calculate_fisher_transform(high, low, close, period=9)
    rsi_14 = calculate_rsi(close, period=14)
    
    # Calculate and align 1w HMA for ultra-macro bias
    hma_1w_raw = calculate_hma(df_1w['close'].values, period=21)
    hma_1w_aligned = align_htf_to_ltf(prices, df_1w, hma_1w_raw)
    
    signals = np.zeros(n)
    
    # Asymmetric position sizing (bear market bias)
    LONG_SIZE = 0.30   # 30% for longs
    SHORT_SIZE = 0.25  # 25% for shorts (prefer shorts in bear)
    
    # Position tracking for stoploss
    in_position = False
    position_side = 0
    entry_price = 0.0
    entry_atr = 0.0
    highest_since_entry = 0.0
    lowest_since_entry = float('inf')
    
    # Fisher Transform tracking for crossover detection
    prev_fisher = 0.0
    
    for i in range(100, n):
        # Skip if indicators not ready
        if np.isnan(atr_14[i]) or atr_14[i] <= 1e-10:
            signals[i] = 0.0
            prev_fisher = fisher[i] if not np.isnan(fisher[i]) else prev_fisher
            continue
        if np.isnan(adx_14[i]) or np.isnan(fisher[i]):
            signals[i] = 0.0
            prev_fisher = fisher[i] if not np.isnan(fisher[i]) else prev_fisher
            continue
        if np.isnan(hma_1w_aligned[i]):
            signals[i] = 0.0
            prev_fisher = fisher[i] if not np.isnan(fisher[i]) else prev_fisher
            continue
        
        # === ULTRA-MACRO BIAS (1w HMA - HARD FILTER) ===
        # Only LONG if price above 1w HMA (weekly bullish)
        # Prefer SHORT if price below 1w HMA (weekly bearish)
        price_above_hma_1w = close[i] > hma_1w_aligned[i]
        price_below_hma_1w = close[i] < hma_1w_aligned[i]
        
        # === TREND STRENGTH (ADX) ===
        is_trending = adx_14[i] > 20.0  # Lower threshold for more trades
        is_weak_trend = adx_14[i] <= 20.0
        
        # === FISHER TRANSFORM SIGNALS ===
        # Long: Fisher crosses above -1.5 (oversold reversal)
        # Short: Fisher crosses below +1.5 (overbought reversal)
        fisher_cross_long = (prev_fisher < -1.5) and (fisher[i] >= -1.5)
        fisher_cross_short = (prev_fisher > 1.5) and (fisher[i] <= 1.5)
        
        # Also allow extreme Fisher values for mean reversion
        fisher_extreme_long = fisher[i] < -2.0
        fisher_extreme_short = fisher[i] > 2.0
        
        # === DESIRED SIGNAL ===
        desired_signal = 0.0
        
        # LONG ENTRY CONDITIONS (asymmetric - need weekly bullish bias)
        if price_above_hma_1w:
            if fisher_cross_long or fisher_extreme_long:
                # Long signal in bullish weekly regime
                if is_trending:
                    desired_signal = LONG_SIZE  # Full size in trend
                else:
                    desired_signal = LONG_SIZE * 0.8  # Reduced in weak trend
        
        # SHORT ENTRY CONDITIONS (prefer shorts in bear market)
        if price_below_hma_1w:
            if fisher_cross_short or fisher_extreme_short:
                # Short signal in bearish weekly regime
                if is_trending:
                    desired_signal = -SHORT_SIZE  # Full size in trend
                else:
                    desired_signal = -SHORT_SIZE * 0.8  # Reduced in weak trend
        
        # === RSI OVERLAY FILTER (avoid extreme overbought/oversold entries) ===
        # Don't long if RSI > 70, don't short if RSI < 30
        if desired_signal > 0 and rsi_14[i] > 70:
            desired_signal = 0.0
        if desired_signal < 0 and rsi_14[i] < 30:
            desired_signal = 0.0
        
        # === STOPLOSS CHECK (2.5 * ATR trailing) ===
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
        
        # === FISHER REVERSAL EXIT ===
        # Exit long when Fisher crosses above +1.5 (overbought)
        # Exit short when Fisher crosses below -1.5 (oversold)
        if in_position and position_side > 0:
            if prev_fisher > 1.5 and fisher[i] <= 1.5:
                desired_signal = 0.0
        
        if in_position and position_side < 0:
            if prev_fisher < -1.5 and fisher[i] >= -1.5:
                desired_signal = 0.0
        
        # === HOLD LOGIC — Maintain position unless exit trigger ===
        if in_position and desired_signal == 0.0 and not stoploss_triggered:
            # Check if bias still valid
            if position_side > 0 and price_above_hma_1w:
                desired_signal = LONG_SIZE * 0.8  # Hold long
            elif position_side < 0 and price_below_hma_1w:
                desired_signal = -SHORT_SIZE * 0.8  # Hold short
        
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
                # Position flip
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
        prev_fisher = fisher[i]
    
    return signals