#!/usr/bin/env python3
"""
Experiment #226: 4h KAMA Adaptive Trend + 1d/1w HMA Regime + RSI Pullback + ADX Filter

Hypothesis: 4h timeframe with KAMA (Kaufman Adaptive Moving Average) adapts to 
volatility regimes better than static EMA/SMA. 1w HMA provides bull/bear market 
regime filter, 1d HMA gives intermediate trend bias. RSI pullback entries (not 
extremes) ensure we enter on dips in established trends. Lower ADX threshold (15) 
ensures sufficient trade count while filtering dead chop.

Why 4h + KAMA might beat current best:
- KAMA adapts smoothing based on volatility ratio (ER) - less lag in trends, 
  more smoothing in ranges
- 1w HMA regime filter prevents counter-trend trades in strong bear markets
- RSI 40-70 range for longs (not 30 oversold) = enter on pullbacks, not crashes
- ADX > 15 (not 25) = enough trades while filtering dead zones
- 2.5*ATR trailing stop protects gains

Learning from failures:
- #219, #221: Fisher transform = 0 trades or massive losses
- #220, #222: Choppiness index regime = negative Sharpe
- #223, #225: Mean reversion on 15m/30m/1h = fails on crypto
- Current best #4h KAMA works - enhance with 1w regime filter

Timeframe: 4h (REQUIRED for this experiment)
HTF: 1d and 1w via mtf_data helper (call ONCE before loop)
Position sizing: 0.30 discrete levels
Stoploss: 2.5 * ATR(14) trailing
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_4h_kama_1d_1w_hma_rsi_adx_atr_v1"
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

def calculate_adx(high, low, close, period=14):
    """Calculate ADX (Average Directional Index) for trend strength."""
    n = len(close)
    
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]
    
    plus_dm = np.zeros(n)
    minus_dm = np.zeros(n)
    
    for i in range(1, n):
        high_diff = high[i] - high[i-1]
        low_diff = low[i-1] - low[i]
        
        if high_diff > low_diff and high_diff > 0:
            plus_dm[i] = high_diff
        if low_diff > high_diff and low_diff > 0:
            minus_dm[i] = low_diff
    
    plus_dm_s = pd.Series(plus_dm).ewm(span=period, min_periods=period, adjust=False).mean().values
    minus_dm_s = pd.Series(minus_dm).ewm(span=period, min_periods=period, adjust=False).mean().values
    tr_s = pd.Series(tr).ewm(span=period, min_periods=period, adjust=False).mean().values
    
    tr_s = np.where(tr_s == 0, 1e-10, tr_s)
    
    plus_di = 100 * plus_dm_s / tr_s
    minus_di = 100 * minus_dm_s / tr_s
    
    dx = 100 * np.abs(plus_di - minus_di) / (plus_di + minus_di + 1e-10)
    adx = pd.Series(dx).ewm(span=period, min_periods=period, adjust=False).mean().values
    
    return adx

def calculate_kama(close, er_period=10, fast_sc=2, slow_sc=30):
    """
    Calculate Kaufman Adaptive Moving Average (KAMA).
    KAMA adapts smoothing based on Efficiency Ratio (ER).
    ER = |net change| / sum of absolute changes over period
    High ER (trending) = fast smoothing, Low ER (choppy) = slow smoothing
    """
    n = len(close)
    kama = np.zeros(n)
    
    # Calculate Efficiency Ratio
    er = np.zeros(n)
    for i in range(er_period, n):
        net_change = np.abs(close[i] - close[i - er_period])
        sum_changes = np.sum(np.abs(np.diff(close[i - er_period:i + 1])))
        if sum_changes > 0:
            er[i] = net_change / sum_changes
        else:
            er[i] = 0
    
    # Calculate smoothing constant
    fast_const = 2.0 / (fast_sc + 1)
    slow_const = 2.0 / (slow_sc + 1)
    
    sc = np.zeros(n)
    for i in range(er_period, n):
        sc[i] = (er[i] * (fast_const - slow_const) + slow_const) ** 2
    
    # Initialize KAMA
    kama[er_period] = close[er_period]
    
    # Calculate KAMA
    for i in range(er_period + 1, n):
        kama[i] = kama[i - 1] + sc[i] * (close[i] - kama[i - 1])
    
    # Fill initial values
    kama[:er_period] = close[:er_period]
    
    return kama

def calculate_hma(close, period=21):
    """Calculate Hull Moving Average for smoother trend with less lag."""
    close_s = pd.Series(close)
    half = max(1, period // 2)
    sqrt_period = max(1, int(np.sqrt(period)))
    wma1 = close_s.ewm(span=half, min_periods=half, adjust=False).mean()
    wma2 = close_s.ewm(span=period, min_periods=period, adjust=False).mean()
    wma3 = (2 * wma1 - wma2).ewm(span=sqrt_period, min_periods=sqrt_period, adjust=False).mean()
    return wma3.values

def calculate_rsi(close, period=14):
    """Calculate RSI (Relative Strength Index)."""
    close_s = pd.Series(close)
    delta = close_s.diff()
    gain = delta.where(delta > 0, 0.0)
    loss = -delta.where(delta < 0, 0.0)
    
    avg_gain = gain.ewm(span=period, min_periods=period, adjust=False).mean()
    avg_loss = loss.ewm(span=period, min_periods=period, adjust=False).mean()
    
    rs = avg_gain / (avg_loss + 1e-10)
    rsi = 100 - (100 / (1 + rs))
    
    return rsi.values

def calculate_ema(close, period=21):
    """Calculate Exponential Moving Average."""
    close_s = pd.Series(close)
    ema = close_s.ewm(span=period, min_periods=period, adjust=False).mean()
    return ema.values

def generate_signals(prices):
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    n = len(close)
    
    # Load HTF data ONCE before loop (Rule 1 - CRITICAL)
    df_1d = get_htf_data(prices, '1d')
    df_1w = get_htf_data(prices, '1w')
    
    # Calculate HTF indicators
    hma_1d = calculate_hma(df_1d['close'].values, 21)
    hma_1w = calculate_hma(df_1w['close'].values, 21)
    
    # Align HTF to LTF (Rule 2 - no manual index mapping, auto shift(1))
    hma_1d_aligned = align_htf_to_ltf(prices, df_1d, hma_1d)
    hma_1w_aligned = align_htf_to_ltf(prices, df_1w, hma_1w)
    
    # Calculate 4h indicators
    atr = calculate_atr(high, low, close, 14)
    adx = calculate_adx(high, low, close, 14)
    kama = calculate_kama(close, er_period=10, fast_sc=2, slow_sc=30)
    ema_50 = calculate_ema(close, 50)
    rsi = calculate_rsi(close, 14)
    
    signals = np.zeros(n)
    
    # Position sizing - discrete levels (Rule 4)
    SIZE_BASE = 0.30
    
    # Track position state for stoploss
    in_position = False
    position_side = 0
    highest_close = 0.0
    lowest_close = 0.0
    
    for i in range(100, n):
        # Skip if indicators not ready
        if np.isnan(atr[i]) or atr[i] == 0:
            signals[i] = 0.0
            continue
        
        if np.isnan(hma_1d_aligned[i]) or np.isnan(hma_1w_aligned[i]):
            signals[i] = 0.0
            continue
        
        if np.isnan(adx[i]) or np.isnan(kama[i]) or np.isnan(rsi[i]):
            signals[i] = 0.0
            continue
        
        # === MULTI-TIMEFRAME REGIME FILTER ===
        # 1w HMA = bull/bear market regime
        bull_regime_1w = close[i] > hma_1w_aligned[i]
        bear_regime_1w = close[i] < hma_1w_aligned[i]
        
        # 1d HMA = intermediate trend bias
        bull_trend_1d = close[i] > hma_1d_aligned[i]
        bear_trend_1d = close[i] < hma_1d_aligned[i]
        
        # === TREND STRENGTH FILTER ===
        # ADX > 15 = trending market (lower threshold for 4h to ensure trades)
        trend_strength = adx[i] > 15
        
        # === KAMA TREND ===
        # Price above KAMA + KAMA above EMA50 = bullish structure
        # Price below KAMA + KAMA below EMA50 = bearish structure
        kama_bullish = close[i] > kama[i] and kama[i] > ema_50[i]
        kama_bearish = close[i] < kama[i] and kama[i] < ema_50[i]
        
        # === RSI PULLBACK (not extreme) ===
        # Long: RSI 40-70 (pullback in uptrend, not crash)
        # Short: RSI 30-60 (rally in downtrend, not squeeze)
        rsi_long_ok = 40 <= rsi[i] <= 70
        rsi_short_ok = 30 <= rsi[i] <= 60
        
        # === PULLBACK ENTRY ===
        # Enter when price pulls back to KAMA (within 2%)
        pullback_long = close[i] <= kama[i] * 1.02
        pullback_short = close[i] >= kama[i] * 0.98
        
        new_signal = 0.0
        
        # === ENTRY CONDITIONS ===
        # Long: 1w bull + 1d bull + ADX trending + KAMA bullish + RSI pullback range
        if bull_regime_1w and bull_trend_1d and trend_strength and kama_bullish:
            if rsi_long_ok and pullback_long:
                new_signal = SIZE_BASE
        
        # Short: 1w bear + 1d bear + ADX trending + KAMA bearish + RSI rally range
        if bear_regime_1w and bear_trend_1d and trend_strength and kama_bearish:
            if rsi_short_ok and pullback_short:
                new_signal = -SIZE_BASE
        
        # === STOPLOSS LOGIC (Rule 6) - 2.5 * ATR trailing ===
        # Check stoploss on EXISTING position before considering new entry
        if in_position:
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
        
        # === UPDATE POSITION TRACKING FOR NEXT BAR ===
        if new_signal != 0.0:
            if not in_position:
                # Entering new position
                in_position = True
                position_side = np.sign(new_signal)
                entry_price = close[i]
                highest_close = close[i] if position_side > 0 else 0.0
                lowest_close = close[i] if position_side < 0 else 0.0
            elif np.sign(new_signal) != position_side:
                # Reversing position
                position_side = np.sign(new_signal)
                entry_price = close[i]
                highest_close = close[i] if position_side > 0 else 0.0
                lowest_close = close[i] if position_side < 0 else 0.0
            # else: maintaining same position direction
        else:
            # Exiting position (signal-based or stoploss)
            if in_position:
                in_position = False
                position_side = 0
                entry_price = 0.0
                highest_close = 0.0
                lowest_close = 0.0
        
        signals[i] = new_signal
    
    return signals