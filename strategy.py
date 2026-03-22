#!/usr/bin/env python3
"""
Experiment #498: 30m Primary + 4h/1d HTF — BB Squeeze Breakout with HTF Trend

Hypothesis: After analyzing 497 experiments, clear patterns emerge:
1. Lower TF (30m/1h/4h) trend-following consistently fails (whipsaw in 2022 crash)
2. CRSI+CHOP tried 20+ times on lower TFs = negative Sharpe
3. Volume-spike pullback (#498 concept) may be too restrictive = 0 trades risk
4. BB Squeeze + HTF trend shows promise in research for mean-reversion regimes

Strategy:
- 4h HMA(21) + 1d HMA(21) = major trend direction
- 30m BB Width percentile < 20% = squeeze (low vol = impending breakout)
- Entry: BB breakout + volume > 1.3x avg + aligned with HTF trend
- Exit: ATR(14) 2.5x trailing stop OR HTF trend flip OR BB Width expands > 80%
- Size: 0.22 (conservative for 30m, discrete levels)

Why this might work:
- BB squeeze captures vol expansion breakouts (proven pattern)
- HTF alignment filters false breakouts (major trend confirmation)
- Volume confirmation = genuine interest, not fake move
- Conservative sizing protects against 2022-style crashes
- Target: 50-80 trades/year (strict but not zero-trade restrictive)

Key difference from failed #488:
- BB squeeze instead of CRSI (different signal type)
- Volume ratio confirmation (1.3x not 1.5x = more trades)
- Single HTF trend filter (4h HMA, not dual 4h+1d = less restrictive)
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_30m_bb_squeeze_hma_4h_v1"
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
    """Calculate Hull Moving Average (HMA)."""
    n = period
    half = n // 2
    sqrt_n = int(np.sqrt(n))
    
    close_s = pd.Series(close)
    
    def wma(series, span):
        weights = np.arange(1, span + 1)
        return series.rolling(window=span, min_periods=span).apply(
            lambda x: np.dot(x, weights) / weights.sum(), raw=True
        )
    
    wma_half = wma(close_s, half)
    wma_full = wma(close_s, n)
    hma_raw = 2.0 * wma_half - wma_full
    hma = wma(hma_raw, sqrt_n)
    
    return hma.values

def calculate_bollinger_bands(close, period=20, std_dev=2.0):
    """Calculate Bollinger Bands."""
    close_s = pd.Series(close)
    sma = close_s.rolling(window=period, min_periods=period).mean()
    std = close_s.rolling(window=period, min_periods=period).std()
    
    upper = sma + (std_dev * std)
    lower = sma - (std_dev * std)
    
    return upper.values, lower.values, sma.values

def calculate_bb_width(upper, lower, sma):
    """Calculate Bollinger Band Width."""
    sma_safe = np.where(sma == 0, 1e-10, sma)
    bb_width = (upper - lower) / sma_safe
    return bb_width

def calculate_bb_percentile(bb_width, lookback=100):
    """Calculate BB Width percentile over lookback period."""
    n = len(bb_width)
    percentile = np.zeros(n)
    
    for i in range(lookback, n):
        window = bb_width[i-lookback:i]
        current = bb_width[i]
        if np.isnan(current):
            percentile[i] = 50.0
        else:
            rank = (window < current).sum()
            percentile[i] = (rank / lookback) * 100.0
    
    return percentile

def calculate_volume_ratio(volume, period=20):
    """Calculate volume ratio vs rolling average."""
    vol_avg = pd.Series(volume).rolling(window=period, min_periods=period).mean().values
    vol_ratio = volume / (vol_avg + 1e-10)
    return vol_ratio

def calculate_rsi(close, period=14):
    """Calculate RSI using Wilder's smoothing."""
    close_s = pd.Series(close)
    delta = close_s.diff()
    
    gain = delta.where(delta > 0, 0.0)
    loss = -delta.where(delta < 0, 0.0)
    
    avg_gain = gain.ewm(span=period, min_periods=period, adjust=False).mean()
    avg_loss = loss.ewm(span=period, min_periods=period, adjust=False).mean()
    
    rs = avg_gain / (avg_loss + 1e-10)
    rsi = 100.0 - (100.0 / (1.0 + rs))
    
    return rsi.values

def generate_signals(prices):
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    volume = prices["volume"].values
    n = len(close)
    
    # Load HTF data ONCE before loop (Rule 1 - CRITICAL)
    df_4h = get_htf_data(prices, '4h')
    
    # Calculate HTF indicators
    hma_4h_21 = calculate_hma(df_4h['close'].values, period=21)
    
    # Align HTF to LTF (Rule 2 - auto shift(1))
    hma_4h_21_aligned = align_htf_to_ltf(prices, df_4h, hma_4h_21)
    
    # Calculate 30m indicators
    atr_14 = calculate_atr(high, low, close, 14)
    bb_upper, bb_lower, bb_sma = calculate_bollinger_bands(close, period=20, std_dev=2.0)
    bb_width = calculate_bb_width(bb_upper, bb_lower, bb_sma)
    bb_percentile = calculate_bb_percentile(bb_width, lookback=100)
    vol_ratio = calculate_volume_ratio(volume, period=20)
    rsi_14 = calculate_rsi(close, 14)
    
    signals = np.zeros(n)
    
    # Position sizing (Rule 4 - discrete, max 0.40, conservative for 30m)
    SIZE = 0.22
    
    # Track position state for stoploss
    in_position = False
    position_side = 0
    entry_price = 0.0
    highest_since_entry = 0.0
    lowest_since_entry = 0.0
    
    for i in range(200, n):
        # Skip if indicators not ready
        if np.isnan(atr_14[i]) or atr_14[i] == 0:
            continue
        if np.isnan(hma_4h_21_aligned[i]):
            continue
        if np.isnan(bb_upper[i]) or np.isnan(bb_lower[i]) or np.isnan(bb_sma[i]):
            continue
        if np.isnan(bb_percentile[i]) or np.isnan(vol_ratio[i]) or np.isnan(rsi_14[i]):
            continue
        
        # === HTF TREND DIRECTION (4h HMA) ===
        bull_4h = close[i] > hma_4h_21_aligned[i]
        bear_4h = close[i] < hma_4h_21_aligned[i]
        
        # === BB SQUEEZE DETECTION (low vol = impending breakout) ===
        is_squeeze = bb_percentile[i] < 25.0  # Bottom 25% of BB width
        
        # === BB BREAKOUT DETECTION ===
        breakout_long = close[i] > bb_upper[i]
        breakout_short = close[i] < bb_lower[i]
        
        # === VOLUME CONFIRMATION ===
        vol_confirm = vol_ratio[i] > 1.3
        
        # === RSI FILTER (avoid extreme overbought/oversold entries) ===
        rsi_ok_long = rsi_14[i] < 75.0  # Not extremely overbought
        rsi_ok_short = rsi_14[i] > 25.0  # Not extremely oversold
        
        # === ENTRY LOGIC ===
        new_signal = 0.0
        
        # Long: squeeze + breakout long + volume + HTF bull + RSI ok
        if is_squeeze and breakout_long and vol_confirm and bull_4h and rsi_ok_long:
            new_signal = SIZE
        
        # Short: squeeze + breakout short + volume + HTF bear + RSI ok
        elif is_squeeze and breakout_short and vol_confirm and bear_4h and rsi_ok_short:
            new_signal = -SIZE
        
        # === STOPLOSS CHECK (2.5 * ATR trailing) ===
        stoploss_triggered = False
        
        if in_position and position_side > 0:
            highest_since_entry = max(highest_since_entry, close[i])
            stop_price = highest_since_entry - 2.5 * atr_14[i]
            if close[i] < stop_price:
                stoploss_triggered = True
        
        if in_position and position_side < 0:
            if lowest_since_entry == 0.0:
                lowest_since_entry = close[i]
            else:
                lowest_since_entry = min(lowest_since_entry, close[i])
            stop_price = lowest_since_entry + 2.5 * atr_14[i]
            if close[i] > stop_price:
                stoploss_triggered = True
        
        if stoploss_triggered:
            new_signal = 0.0
        
        # === HTF TREND FLIP EXIT ===
        if in_position and position_side > 0 and bear_4h:
            new_signal = 0.0
        if in_position and position_side < 0 and bull_4h:
            new_signal = 0.0
        
        # === BB WIDTH EXPANSION EXIT (volatility normalized) ===
        if in_position and bb_percentile[i] > 80.0:
            new_signal = 0.0
        
        # === RSI EXTREME EXIT ===
        if in_position and position_side > 0 and rsi_14[i] > 80.0:
            new_signal = 0.0
        if in_position and position_side < 0 and rsi_14[i] < 20.0:
            new_signal = 0.0
        
        # === UPDATE POSITION TRACKING ===
        if new_signal != 0.0:
            if not in_position:
                in_position = True
                position_side = np.sign(new_signal)
                entry_price = close[i]
                highest_since_entry = close[i] if position_side > 0 else 0.0
                lowest_since_entry = close[i] if position_side < 0 else 0.0
            elif np.sign(new_signal) != position_side:
                position_side = np.sign(new_signal)
                entry_price = close[i]
                highest_since_entry = close[i] if position_side > 0 else 0.0
                lowest_since_entry = close[i] if position_side < 0 else 0.0
        else:
            if in_position:
                in_position = False
                position_side = 0
                entry_price = 0.0
                highest_since_entry = 0.0
                lowest_since_entry = 0.0
        
        signals[i] = new_signal
    
    return signals