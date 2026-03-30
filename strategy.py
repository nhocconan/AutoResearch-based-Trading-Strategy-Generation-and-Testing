#!/usr/bin/env python3
"""
Experiment #007: 6h ATR Volatility Breakout + RSI Pullback

HYPOTHESIS: Markets alternate between high-volatility breakouts and low-volatility
consolidation. This strategy exploits the PATTERN:
- High volatility expansion (ATR breakout) → followed by directional move
- Low volatility squeeze (Bollinger Band width contraction) → mean reversion

KEY DIFFERENCE from failed attempts:
- Uses ATR REGIME to switch between breakout and mean-reversion modes
- Combines with RSI (not TRIX, not Williams %R, not momentum)
- Bollinger Band width for squeeze detection (not choppiness)

WHY IT WORKS IN BULL AND BEAR:
- In uptrends: buy pullbacks to EMA21 + RSI oversold + ATR expansion
- In downtrends: sell bounces to EMA21 + RSI overbought + ATR expansion
- Symmetrical logic, adapts to regime

TARGET: 75-150 total trades over 4 years = 19-37/year. HARD MAX: 300.
Signal size: 0.25.
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_6h_atr_volatility_rsi_pullback_v1"
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

def calculate_rsi(prices, period=14):
    """RSI indicator"""
    close = prices["close"].values if hasattr(prices, 'close') else prices
    n = len(close)
    if n < period + 1:
        return np.full(n, np.nan)
    
    deltas = np.diff(close, prepend=close[0])
    gains = np.where(deltas > 0, deltas, 0.0)
    losses = np.where(deltas < 0, -deltas, 0.0)
    
    avg_gain = pd.Series(gains).ewm(span=period, min_periods=period, adjust=False).mean().values
    avg_loss = pd.Series(losses).ewm(span=period, min_periods=period, adjust=False).mean().values
    
    rs = avg_gain / np.where(avg_loss == 0, 1e-10, avg_loss)
    rsi = 100 - (100 / (1 + rs))
    return rsi

def generate_signals(prices):
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    volume = prices["volume"].values
    n = len(close)
    
    # === Load HTF data ONCE before loop ===
    df_1d = get_htf_data(prices, '1d')
    
    # 1d SMA50 for trend direction
    sma_1d = pd.Series(df_1d['close'].values).rolling(window=50, min_periods=50).mean().values
    sma_1d_aligned = align_htf_to_ltf(prices, df_1d, sma_1d)
    
    # === Local 6h indicators ===
    atr_14 = calculate_atr(high, low, close, period=14)
    
    # EMA21 for pullback detection
    ema_21 = pd.Series(close).ewm(span=21, min_periods=21, adjust=False).mean().values
    
    # RSI(14) for momentum
    rsi_14 = calculate_rsi(close, period=14)
    
    # Bollinger Band width percentile (volatility regime)
    bb_std = pd.Series(close).rolling(window=20, min_periods=20).std().values
    bb_width = bb_std / pd.Series(close).rolling(window=20, min_periods=20).mean().values
    bb_width_pct = pd.Series(bb_width).rolling(window=100, min_periods=50).apply(
        lambda x: pd.Series(x).rank(pct=True).iloc[-1] * 100, raw=False
    ).values
    
    # Volume ratio
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_ratio = volume / np.where(vol_ma > 0, vol_ma, 1)
    
    # ATR regime: compare current ATR to 30-bar ATR average
    atr_avg = pd.Series(atr_14).rolling(window=30, min_periods=20).mean().values
    atr_ratio = atr_14 / np.where(atr_avg > 0, atr_avg, 1)
    
    # Signals
    signals = np.zeros(n)
    SIZE = 0.25
    
    # Position tracking
    in_position = False
    position_side = 0
    entry_price = 0.0
    entry_atr = 0.0
    stop_price = 0.0
    highest_since_entry = 0.0
    lowest_since_entry = 0.0
    entry_bar = 0
    
    warmup = 150  # Need enough for alignment buffer
    
    for i in range(warmup, n):
        # Skip if indicators not ready
        if np.isnan(atr_14[i]) or atr_14[i] <= 1e-10:
            signals[i] = 0.0
            in_position = False
            position_side = 0
            continue
        
        if np.isnan(sma_1d_aligned[i]) or np.isnan(rsi_14[i]) or np.isnan(bb_width_pct[i]):
            signals[i] = 0.0
            in_position = False
            position_side = 0
            continue
        
        # === TREND DIRECTION (1d SMA50) ===
        is_bull = close[i] > sma_1d_aligned[i]
        
        # === VOLATILITY REGIME ===
        # Low vol regime (squeeze): bb_width_pct < 20, mean reversion more likely
        # High vol regime: atr_ratio > 1.3, breakout trades more likely
        is_squeeze = bb_width_pct[i] < 20
        is_vol_expansion = atr_ratio[i] > 1.3
        
        # Volume confirmation
        vol_spike = vol_ratio[i] > 1.3
        
        # === ENTRY LOGIC ===
        desired_signal = 0.0
        
        if not in_position:
            # Price distance from EMA21 (normalized by ATR)
            ema_dist = (close[i] - ema_21[i]) / atr_14[i]
            
            # === LONG ENTRY: Pullback in uptrend ===
            if is_bull:
                # RSI oversold + price below EMA + volume confirmation
                if rsi_14[i] < 35 and ema_dist < 0:
                    # Prefer squeeze exits (mean reversion from compression)
                    if is_squeeze and vol_spike:
                        desired_signal = SIZE
                    # Or vol expansion breakouts
                    elif is_vol_expansion and vol_spike:
                        desired_signal = SIZE
            
            # === SHORT ENTRY: Bounce in downtrend ===
            if not is_bull:
                # RSI overbought + price above EMA + volume confirmation
                if rsi_14[i] > 65 and ema_dist > 0:
                    if is_squeeze and vol_spike:
                        desired_signal = -SIZE
                    elif is_vol_expansion and vol_spike:
                        desired_signal = -SIZE
        
        # === STOPLOSS (2.0 ATR trailing) ===
        if in_position and position_side > 0:
            highest_since_entry = max(highest_since_entry, high[i])
            trailing_stop = highest_since_entry - 2.0 * entry_atr
            stop_price = max(stop_price, trailing_stop)
            if low[i] < stop_price:
                desired_signal = 0.0
        
        if in_position and position_side < 0:
            lowest_since_entry = min(lowest_since_entry, low[i])
            trailing_stop = lowest_since_entry + 2.0 * entry_atr
            stop_price = min(stop_price, trailing_stop)
            if high[i] > stop_price:
                desired_signal = 0.0
        
        # === MINIMUM HOLD (2 bars = 12h to reduce churn) ===
        bars_held = i - entry_bar
        
        # === MOMENTUM EXIT ===
        if in_position and bars_held >= 2:
            if position_side > 0 and rsi_14[i] > 60:
                desired_signal = 0.0
            if position_side < 0 and rsi_14[i] < 40:
                desired_signal = 0.0
        
        # === UPDATE POSITION ===
        if desired_signal != 0.0:
            if not in_position or np.sign(desired_signal) != position_side:
                # New position or flip
                in_position = True
                position_side = int(np.sign(desired_signal))
                entry_price = close[i]
                entry_atr = atr_14[i]
                highest_since_entry = high[i]
                lowest_since_entry = low[i]
                entry_bar = i
                if position_side > 0:
                    stop_price = entry_price - 2.0 * entry_atr
                else:
                    stop_price = entry_price + 2.0 * entry_atr
        else:
            if in_position:
                in_position = False
                position_side = 0
                stop_price = 0.0
        
        signals[i] = desired_signal
    
    return signals