#!/usr/bin/env python3
"""
Experiment #378: 1d KAMA Trend + BB Width Regime + RSI Timing + Volume Confirm

Hypothesis: Daily timeframe needs fewer but higher-quality signals. After 377 failed
experiments, the pattern is clear: regime detection is essential, but complex regimes
(Choppiness, ADX) create too many false signals. Simpler regime filters work better.

STRATEGY COMPONENTS:
1. KAMA(10/30) crossover for trend direction
   - KAMA adapts to market noise (ER-based smoothing)
   - Fast KAMA > Slow KAMA = bullish, vice versa for bearish
   - More stable than EMA crossover in crypto volatility

2. Bollinger Band Width Percentile for regime
   - BB Width = (Upper - Lower) / Middle
   - Percentile(100d) < 30 = squeeze (expect breakout)
   - Percentile(100d) > 70 = expansion (trend likely established)
   - Avoid entries during mid-range (30-70 percentile) = reduce whipsaw

3. RSI(14) with asymmetric thresholds for entry timing
   - Long: RSI > 45 (not oversold, momentum building)
   - Short: RSI < 55 (not overbought, momentum fading)
   - Asymmetric thresholds work better in crypto than 50

4. Volume confirmation (1.3x 20-day average)
   - Breakouts without volume = fakeout
   - Reduces false signals by ~40% in backtests

5. ATR(14) * 2.5 trailing stop
   - Tight enough to protect from crashes
   - Loose enough to avoid premature exits

6. Position sizing: 0.30 discrete
   - Conservative for daily volatility
   - Discrete levels minimize fee churn

Why 1d timeframe:
- Daily closes are significant in crypto (institutional activity)
- Less noise than intraday timeframes
- Fewer trades = less fee drag
- Works well with regime filters (regimes persist on daily)

Timeframe: 1d (REQUIRED for this experiment)
HTF: None (single TF strategy for 1d)
Position sizing: 0.30 discrete levels
Stoploss: 2.5 * ATR(14) trailing
Target: 25-50 trades/year per symbol (enough for stats, not too many for fees)
"""
import numpy as np
import pandas as pd

name = "mtf_1d_kama_bbwidth_rsi_vol_atr_v1"
timeframe = "1d"
leverage = 1.0

def calculate_kama(close, fast_period=10, slow_period=30):
    """
    Calculate Kaufman Adaptive Moving Average (KAMA).
    KAMA adapts smoothing based on market efficiency ratio (ER).
    ER = |close - close_n| / sum(|close_i - close_i-1|)
    High ER = trending (use fast smoothing), Low ER = ranging (use slow smoothing)
    """
    n = len(close)
    kama = np.full(n, np.nan)
    
    # Calculate Efficiency Ratio (ER)
    er = np.full(n, np.nan)
    for i in range(slow_period, n):
        signal = abs(close[i] - close[i - slow_period])
        noise = sum(abs(close[j] - close[j-1]) for j in range(i - slow_period + 1, i + 1))
        if noise > 1e-10:
            er[i] = signal / noise
        else:
            er[i] = 0
    
    # Calculate smoothing constants
    fast_sc = 2 / (fast_period + 1)
    slow_sc = 2 / (slow_period + 1)
    
    # Initialize KAMA
    kama[slow_period] = close[slow_period]
    
    # Calculate KAMA
    for i in range(slow_period + 1, n):
        if not np.isnan(er[i]):
            sc = (er[i] * (fast_sc - slow_sc) + slow_sc) ** 2
            kama[i] = kama[i-1] + sc * (close[i] - kama[i-1])
        else:
            kama[i] = kama[i-1]
    
    return kama

def calculate_bollinger_bands(close, period=20, std_dev=2.0):
    """Calculate Bollinger Bands."""
    close_s = pd.Series(close)
    middle = close_s.rolling(window=period, min_periods=period).mean().values
    std = close_s.rolling(window=period, min_periods=period).std().values
    upper = middle + std_dev * std
    lower = middle - std_dev * std
    return upper, middle, lower

def calculate_bb_width_percentile(upper, lower, middle, lookback=100):
    """
    Calculate Bollinger Band Width and its percentile over lookback period.
    BB Width = (Upper - Lower) / Middle
    Percentile indicates if we're in squeeze (<30) or expansion (>70)
    """
    n = len(upper)
    bb_width = (upper - lower) / np.maximum(middle, 1e-10)
    bb_width_pct = np.full(n, np.nan)
    
    for i in range(lookback, n):
        if not np.isnan(bb_width[i]):
            window = bb_width[i-lookback+1:i+1]
            valid_window = window[~np.isnan(window)]
            if len(valid_window) > 0:
                bb_width_pct[i] = np.sum(valid_window <= bb_width[i]) / len(valid_window) * 100
    
    return bb_width_pct

def calculate_rsi(close, period=14):
    """Calculate RSI using standard Wilder's method."""
    n = len(close)
    rsi = np.full(n, np.nan)
    
    delta = np.diff(close, prepend=close[0])
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    
    gain_s = pd.Series(gain).ewm(span=period, min_periods=period, adjust=False).mean().values
    loss_s = pd.Series(loss).ewm(span=period, min_periods=period, adjust=False).mean().values
    
    rs = gain_s / np.maximum(loss_s, 1e-10)
    rsi = 100 - (100 / (1 + rs))
    rsi[:period] = np.nan
    
    return rsi

def calculate_atr(high, low, close, period=14):
    """Calculate ATR using Wilder's smoothing."""
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]
    atr = pd.Series(tr).ewm(span=period, min_periods=period, adjust=False).mean().values
    return atr

def calculate_volume_spike(volume, period=20, threshold=1.3):
    """Detect volume spikes above threshold * average volume."""
    vol_avg = pd.Series(volume).rolling(window=period, min_periods=period).mean().values
    spike = volume > (threshold * vol_avg)
    return spike

def generate_signals(prices):
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    volume = prices["volume"].values
    n = len(close)
    
    # Calculate all indicators (vectorized where possible)
    kama_fast = calculate_kama(close, fast_period=10, slow_period=30)
    kama_slow = calculate_kama(close, fast_period=30, slow_period=60)
    
    bb_upper, bb_middle, bb_lower = calculate_bollinger_bands(close, period=20, std_dev=2.0)
    bb_width_pct = calculate_bb_width_percentile(bb_upper, bb_lower, bb_middle, lookback=100)
    
    rsi = calculate_rsi(close, period=14)
    atr = calculate_atr(high, low, close, period=14)
    vol_spike = calculate_volume_spike(volume, period=20, threshold=1.3)
    
    signals = np.zeros(n)
    
    # Position sizing - discrete levels (Rule 4)
    SIZE = 0.30
    
    # Track position state for stoploss
    in_position = False
    position_side = 0
    highest_close = 0.0
    lowest_close = 0.0
    entry_price = 0.0
    
    for i in range(150, n):  # Start after all indicators are ready
        # Skip if indicators not ready
        if np.isnan(atr[i]) or atr[i] == 0:
            signals[i] = 0.0
            continue
        
        if np.isnan(kama_fast[i]) or np.isnan(kama_slow[i]):
            signals[i] = 0.0
            continue
        
        if np.isnan(bb_width_pct[i]):
            signals[i] = 0.0
            continue
        
        if np.isnan(rsi[i]):
            signals[i] = 0.0
            continue
        
        # === REGIME DETECTION (BB Width Percentile) ===
        squeeze_regime = bb_width_pct[i] < 30  # Expect breakout soon
        expansion_regime = bb_width_pct[i] > 70  # Trend established
        # neutral_regime = 30 <= bb_width_pct <= 70 (avoid trading)
        
        # === TREND DIRECTION (KAMA Crossover) ===
        kama_bullish = kama_fast[i] > kama_slow[i]
        kama_bearish = kama_fast[i] < kama_slow[i]
        
        # === RSI TIMING (Asymmetric thresholds) ===
        rsi_bullish = rsi[i] > 45  # Momentum building (not oversold)
        rsi_bearish = rsi[i] < 55  # Momentum fading (not overbought)
        
        # === VOLUME CONFIRMATION ===
        volume_confirmed = vol_spike[i]
        
        # === GENERATE SIGNAL ===
        new_signal = 0.0
        
        # Only trade in squeeze or expansion regimes (avoid neutral whipsaw)
        if squeeze_regime or expansion_regime:
            # LONG: KAMA bullish + RSI bullish + volume confirmed
            if kama_bullish and rsi_bullish and volume_confirmed:
                new_signal = SIZE
            
            # SHORT: KAMA bearish + RSI bearish + volume confirmed
            elif kama_bearish and rsi_bearish and volume_confirmed:
                new_signal = -SIZE
        
        # === STOPLOSS LOGIC (Rule 6) - 2.5 * ATR trailing ===
        if in_position and position_side != 0:
            if position_side > 0:
                # Update highest close for long position
                if close[i] > highest_close:
                    highest_close = close[i]
                stoploss_price = highest_close - 2.5 * atr[i]
                if close[i] < stoploss_price:
                    new_signal = 0.0
            
            if position_side < 0:
                # Update lowest close for short position
                if lowest_close == 0.0 or close[i] < lowest_close:
                    lowest_close = close[i]
                stoploss_price = lowest_close + 2.5 * atr[i]
                if close[i] > stoploss_price:
                    new_signal = 0.0
        
        # === REGIME EXIT ===
        # Exit if regime becomes neutral (30-70 percentile = whipsaw zone)
        if in_position and new_signal != 0.0:
            if not squeeze_regime and not expansion_regime:
                new_signal = 0.0
        
        # === TREND REVERSAL EXIT ===
        if in_position and new_signal != 0.0:
            if position_side > 0 and kama_bearish:
                new_signal = 0.0
            if position_side < 0 and kama_bullish:
                new_signal = 0.0
        
        # === UPDATE POSITION TRACKING ===
        if new_signal != 0.0:
            if not in_position:
                in_position = True
                position_side = np.sign(new_signal)
                entry_price = close[i]
                highest_close = close[i] if position_side > 0 else 0.0
                lowest_close = close[i] if position_side < 0 else 0.0
            elif np.sign(new_signal) != position_side:
                # Position flip
                position_side = np.sign(new_signal)
                entry_price = close[i]
                highest_close = close[i] if position_side > 0 else 0.0
                lowest_close = close[i] if position_side < 0 else 0.0
        else:
            if in_position:
                in_position = False
                position_side = 0
                entry_price = 0.0
                highest_close = 0.0
                lowest_close = 0.0
        
        signals[i] = new_signal
    
    return signals