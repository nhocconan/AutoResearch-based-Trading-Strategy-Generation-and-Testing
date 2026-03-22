#!/usr/bin/env python3
"""
Experiment #488: 30m Volatility Squeeze Breakout with 4h Trend Bias

Hypothesis: After 12 consecutive failures with complex regime-switching logic,
the issue is OVER-FILTERING. Strategies with Choppiness Index, multiple RSI
thresholds, and asymmetric regime logic generate too few trades or exit too early.

This strategy uses SIMPLER logic:
1. VOLATILITY SQUEEZE DETECTION: Bollinger Band Width at 60-bar percentile < 30
   - Identifies compression before expansion (classic VCP pattern)
2. BREAKOUT ENTRY: Price closes above/below Donchian(20) with volume confirmation
3. 4H TREND BIAS: Only trade breakout in direction of 4h HMA(21) trend
   - Simpler than ADX/Choppiness regime filters
4. ATR(14) STOPLOSS at 2.2x: Tighter than previous 3.0x to reduce drawdown
5. POSITION SIZING: 0.25 discrete (conservative after recent -79% losses)

Why this should work on 30m:
- Volatility squeeze patterns work across all timeframes (30m has enough bars)
- Fewer filters = more trades (addresses #1 failure mode: 0 trades)
- 4h HMA is simpler than Choppiness + ADX + RSI regime combination
- Tighter stops (2.2x vs 3.0x) should reduce max drawdown
- Volume confirmation filters false breakouts

Timeframe: 30m (REQUIRED for this experiment)
HTF: 4h via mtf_data helper (call ONCE before loop)
Position sizing: 0.25 discrete levels
Stoploss: 2.2 * ATR(14) trailing
Target: 50-100 trades/year per symbol (enough for Sharpe, not too many for fees)
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_30m_vol_squeeze_4h_hma_donchian_volume_atr_v1"
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

def calculate_bollinger_bands(close, period=20, std_dev=2.0):
    """Calculate Bollinger Bands."""
    close_s = pd.Series(close)
    sma = close_s.rolling(window=period, min_periods=period).mean()
    std = close_s.rolling(window=period, min_periods=period).std()
    upper = sma + (std_dev * std)
    lower = sma - (std_dev * std)
    return upper.values, lower.values, sma.values

def calculate_bb_width(upper, lower, sma):
    """Calculate Bollinger Band Width as percentage."""
    width = (upper - lower) / sma
    return width

def calculate_percentile_rank(series, window=60):
    """Calculate percentile rank of current value over rolling window."""
    n = len(series)
    pr = np.full(n, np.nan)
    for i in range(window, n):
        if not np.isnan(series[i]):
            window_vals = series[i-window+1:i+1]
            window_vals = window_vals[~np.isnan(window_vals)]
            if len(window_vals) > 0:
                pr[i] = np.sum(window_vals < series[i]) / len(window_vals)
    return pr

def calculate_donchian(high, low, period=20):
    """Calculate Donchian Channel."""
    n = len(high)
    upper = np.full(n, np.nan)
    lower = np.full(n, np.nan)
    
    for i in range(period - 1, n):
        upper[i] = high[i-period+1:i+1].max()
        lower[i] = low[i-period+1:i+1].min()
    
    return upper, lower

def calculate_rsi(close, period=14):
    """Calculate Relative Strength Index."""
    close_s = pd.Series(close)
    delta = close_s.diff()
    gain = delta.where(delta > 0, 0.0)
    loss = -delta.where(delta < 0, 0.0)
    
    avg_gain = gain.ewm(span=period, min_periods=period, adjust=False).mean()
    avg_loss = loss.ewm(span=period, min_periods=period, adjust=False).mean()
    
    rs = avg_gain / avg_loss.replace(0, np.inf)
    rsi = 100 - (100 / (1 + rs))
    return rsi.values

def calculate_volume_sma(volume, period=20):
    """Calculate Volume Simple Moving Average."""
    vol_s = pd.Series(volume)
    return vol_s.rolling(window=period, min_periods=period).mean().values

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
    
    # Align HTF to LTF (Rule 2 - auto shift(1) for completed bars)
    hma_4h_aligned = align_htf_to_ltf(prices, df_4h, hma_4h)
    
    # Calculate 30m indicators
    atr = calculate_atr(high, low, close, 14)
    bb_upper, bb_lower, bb_sma = calculate_bollinger_bands(close, 20, 2.0)
    bb_width = calculate_bb_width(bb_upper, bb_lower, bb_sma)
    bb_width_pr = calculate_percentile_rank(bb_width, 60)
    donchian_upper, donchian_lower = calculate_donchian(high, low, 20)
    rsi = calculate_rsi(close, 14)
    vol_sma = calculate_volume_sma(volume, 20)
    
    signals = np.zeros(n)
    
    # Position sizing - discrete levels (Rule 4)
    SIZE = 0.25
    
    # Track position state for stoploss
    in_position = False
    position_side = 0
    entry_price = 0.0
    highest_close = 0.0
    lowest_close = 0.0
    
    for i in range(100, n):
        # Skip if indicators not ready
        if np.isnan(atr[i]) or atr[i] == 0:
            signals[i] = 0.0
            continue
        
        if np.isnan(hma_4h_aligned[i]):
            signals[i] = 0.0
            continue
        
        if np.isnan(bb_width_pr[i]) or np.isnan(donchian_upper[i]) or np.isnan(donchian_lower[i]):
            signals[i] = 0.0
            continue
        
        if np.isnan(rsi[i]) or np.isnan(vol_sma[i]) or vol_sma[i] == 0:
            signals[i] = 0.0
            continue
        
        # === 4H TREND BIAS ===
        # Simple: price above 4h HMA = bullish bias, below = bearish bias
        bullish_bias = close[i] > hma_4h_aligned[i]
        bearish_bias = close[i] < hma_4h_aligned[i]
        
        # === VOLATILITY SQUEEZE DETECTION ===
        # BB Width percentile < 30 = volatility compression (squeeze)
        squeeze_active = bb_width_pr[i] < 0.30
        
        # === VOLUME CONFIRMATION ===
        volume_confirmed = volume[i] > 1.5 * vol_sma[i]
        
        # === BREAKOUT ENTRY LOGIC ===
        new_signal = 0.0
        
        # LONG: Squeeze + Bullish 4h bias + Donchian breakout + Volume
        if squeeze_active and bullish_bias:
            if close[i] > donchian_upper[i-1] if i > 0 else False:
                if volume_confirmed and rsi[i] > 50:
                    new_signal = SIZE
        
        # SHORT: Squeeze + Bearish 4h bias + Donchian breakdown + Volume
        if squeeze_active and bearish_bias:
            if close[i] < donchian_lower[i-1] if i > 0 else False:
                if volume_confirmed and rsi[i] < 50:
                    new_signal = -SIZE
        
        # === STOPLOSS LOGIC (Rule 6) - 2.2 * ATR trailing ===
        if in_position and position_side != 0:
            if position_side > 0:
                # Update highest close for long position
                if close[i] > highest_close:
                    highest_close = close[i]
                stoploss_price = highest_close - 2.2 * atr[i]
                if close[i] < stoploss_price:
                    new_signal = 0.0
            
            if position_side < 0:
                # Update lowest close for short position
                if lowest_close == 0.0 or close[i] < lowest_close:
                    lowest_close = close[i]
                stoploss_price = lowest_close + 2.2 * atr[i]
                if close[i] > stoploss_price:
                    new_signal = 0.0
        
        # === TREND REVERSAL EXIT ===
        # Exit if 4h trend flips against position
        if in_position and new_signal != 0.0:
            if position_side > 0 and bearish_bias:
                new_signal = 0.0
            if position_side < 0 and bullish_bias:
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