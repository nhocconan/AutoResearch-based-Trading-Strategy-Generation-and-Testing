#!/usr/bin/env python3
"""
Experiment #236: 30m Fisher Transform + KAMA Trend + 4h HMA + ADX Regime + ATR Stop

Hypothesis: 30m timeframe captures intraday swings while 4h HMA provides stable trend bias.
Fisher Transform excels at identifying reversals in bear/range markets (2025 conditions).
KAMA adapts to volatility - fast in trends, slow in ranges. ADX regime filter switches
between trend-following (ADX>25) and mean-reversion (ADX<20) logic. This adaptive
approach should work better than pure trend or pure mean-reversion strategies that
have failed recently.

Why 30m + Fisher might work:
- 30m = 48 bars/day, captures intraday reversals without 15m noise
- Fisher Transform normalizes price to Gaussian distribution, extreme values = reversals
- KAMA (Kaufman Adaptive) reduces whipsaws in choppy markets
- 4h HMA filter prevents counter-trend trades
- ADX regime switching adapts to market conditions (trend vs range)
- Conservative sizing (0.25) controls drawdown

Learning from failures:
- #224 (30m Supertrend): Sharpe=0.000 - pure trend failed
- #230 (30m KAMA+BB): Sharpe=-0.582 - regime filter too strict
- #235 (30m RSI+Chop): Sharpe=-3.648 - too many conflicting filters
- Mean reversion alone fails, trend alone fails - need adaptive regime
- Fisher Transform has shown promise for bear market reversals

Timeframe: 30m (REQUIRED for this experiment)
HTF: 4h via mtf_data helper (call ONCE before loop)
Position sizing: 0.25 discrete levels
Stoploss: 2.0 * ATR(14) trailing
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_30m_fisher_kama_4h_hma_adx_regime_atr_v1"
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
    """Calculate Kaufman Adaptive Moving Average."""
    n = len(close)
    kama = np.zeros(n)
    
    # Calculate Efficiency Ratio
    er = np.zeros(n)
    for i in range(er_period, n):
        change = np.abs(close[i] - close[i - er_period])
        noise = np.sum(np.abs(np.diff(close[i - er_period:i + 1])))
        if noise > 0:
            er[i] = change / noise
        else:
            er[i] = 0
    
    # Calculate Smoothing Constant
    sc = (er * (2 / (fast_sc + 1) - 2 / (slow_sc + 1)) + 2 / (slow_sc + 1)) ** 2
    
    # Initialize KAMA
    kama[er_period] = close[er_period]
    
    # Calculate KAMA
    for i in range(er_period + 1, n):
        kama[i] = kama[i - 1] + sc[i] * (close[i] - kama[i - 1])
    
    # Fill initial values
    kama[:er_period] = close[:er_period].mean()
    
    return kama

def calculate_fisher(close, period=9):
    """Calculate Ehlers Fisher Transform for reversal detection."""
    n = len(close)
    fisher = np.zeros(n)
    trigger = np.zeros(n)
    
    # Calculate highest high and lowest low over period
    hh = np.zeros(n)
    ll = np.zeros(n)
    
    for i in range(period - 1, n):
        hh[i] = np.max(close[i - period + 1:i + 1])
        ll[i] = np.min(close[i - period + 1:i + 1])
    
    hh[:period - 1] = hh[period - 1]
    ll[:period - 1] = ll[period - 1]
    
    # Calculate normalized price
    norm = np.zeros(n)
    for i in range(period - 1, n):
        if hh[i] > ll[i]:
            norm[i] = 0.999 * (close[i] - ll[i]) / (hh[i] - ll[i]) - 0.5
        else:
            norm[i] = 0
    
    # Calculate Fisher Transform
    for i in range(period - 1, n):
        if np.abs(norm[i]) < 0.999:
            fisher[i] = 0.5 * np.log((1 + norm[i]) / (1 - norm[i]))
        else:
            fisher[i] = fisher[i - 1] if i > 0 else 0
    
    # Trigger line (1-period lag of Fisher)
    trigger[1:] = fisher[:-1]
    
    # Fill initial values
    fisher[:period - 1] = 0
    trigger[:period - 1] = 0
    
    return fisher, trigger

def calculate_bollinger(close, period=20, std_dev=2.0):
    """Calculate Bollinger Bands."""
    close_s = pd.Series(close)
    sma = close_s.rolling(window=period, min_periods=period).mean().values
    std = close_s.rolling(window=period, min_periods=period).std().values
    
    upper = sma + std_dev * std
    lower = sma - std_dev * std
    
    # Fill initial values
    upper[:period - 1] = np.nan
    lower[:period - 1] = np.nan
    
    return upper, lower, sma

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

def generate_signals(prices):
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    n = len(close)
    
    # Load HTF data ONCE before loop (Rule 1 - CRITICAL)
    df_4h = get_htf_data(prices, '4h')
    
    # Calculate HTF indicators
    hma_4h = calculate_hma(df_4h['close'].values, 21)
    
    # Align HTF to LTF (Rule 2 - no manual index mapping, auto shift(1))
    hma_4h_aligned = align_htf_to_ltf(prices, df_4h, hma_4h)
    
    # Calculate 30m indicators
    atr = calculate_atr(high, low, close, 14)
    adx = calculate_adx(high, low, close, 14)
    kama = calculate_kama(close, er_period=10)
    fisher, fisher_trigger = calculate_fisher(close, period=9)
    bb_upper, bb_lower, bb_mid = calculate_bollinger(close, period=20, std_dev=2.0)
    rsi = calculate_rsi(close, 14)
    
    signals = np.zeros(n)
    
    # Position sizing - discrete levels (Rule 4)
    SIZE_BASE = 0.25
    
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
        
        if np.isnan(hma_4h_aligned[i]):
            signals[i] = 0.0
            continue
        
        if np.isnan(adx[i]) or np.isnan(kama[i]) or np.isnan(fisher[i]):
            signals[i] = 0.0
            continue
        
        if np.isnan(bb_upper[i]) or np.isnan(bb_lower[i]):
            signals[i] = 0.0
            continue
        
        # === MULTI-TIMEFRAME TREND BIAS ===
        # 4h HMA = higher timeframe trend bias
        bull_trend_4h = close[i] > hma_4h_aligned[i]
        bear_trend_4h = close[i] < hma_4h_aligned[i]
        
        # === REGIME DETECTION ===
        # ADX > 25 = trending market (use trend-following logic)
        # ADX < 20 = ranging market (use mean-reversion logic)
        # ADX 20-25 = transition (no new entries)
        trending_regime = adx[i] > 25
        ranging_regime = adx[i] < 20
        
        # === KAMA TREND ===
        # Price above KAMA = bullish
        # Price below KAMA = bearish
        kama_bullish = close[i] > kama[i]
        kama_bearish = close[i] < kama[i]
        
        # === FISHER TRANSFORM REVERSAL ===
        # Fisher crosses above -1.5 from below = long reversal signal
        # Fisher crosses below +1.5 from above = short reversal signal
        fisher_long_signal = fisher[i] > -1.5 and fisher_trigger[i] <= -1.5
        fisher_short_signal = fisher[i] < 1.5 and fisher_trigger[i] >= 1.5
        
        # === BOLLINGER BAND MEAN REVERSION ===
        # Price at lower band = oversold (long opportunity in range)
        # Price at upper band = overbought (short opportunity in range)
        bb_long_signal = close[i] <= bb_lower[i] * 1.002  # Within 0.2% of lower band
        bb_short_signal = close[i] >= bb_upper[i] * 0.998  # Within 0.2% of upper band
        
        # === RSI CONFIRMATION ===
        rsi_oversold = rsi[i] < 35
        rsi_overbought = rsi[i] > 65
        
        new_signal = 0.0
        
        # === TRENDING REGIME ENTRIES (ADX > 25) ===
        if trending_regime:
            # Long: 4h bullish + KAMA bullish + Fisher reversal confirmation
            if bull_trend_4h and kama_bullish and fisher_long_signal:
                new_signal = SIZE_BASE
            
            # Short: 4h bearish + KAMA bearish + Fisher reversal confirmation
            if bear_trend_4h and kama_bearish and fisher_short_signal:
                new_signal = -SIZE_BASE
        
        # === RANGING REGIME ENTRIES (ADX < 20) ===
        if ranging_regime:
            # Long: 4h bullish bias + BB lower + RSI oversold
            if bull_trend_4h and bb_long_signal and rsi_oversold:
                new_signal = SIZE_BASE
            
            # Short: 4h bearish bias + BB upper + RSI overbought
            if bear_trend_4h and bb_short_signal and rsi_overbought:
                new_signal = -SIZE_BASE
        
        # === STOPLOSS LOGIC (Rule 6) - 2.0 * ATR trailing ===
        # Check stoploss on EXISTING position before considering new entry
        if in_position:
            if position_side > 0:
                # Update highest close for long position
                if close[i] > highest_close:
                    highest_close = close[i]
                # Trailing stop: 2.0 * ATR below highest close
                stoploss_price = highest_close - 2.0 * atr[i]
                if close[i] < stoploss_price:
                    new_signal = 0.0  # Stoploss overrides entry signal
            
            if position_side < 0:
                # Update lowest close for short position
                if lowest_close == 0.0 or close[i] < lowest_close:
                    lowest_close = close[i]
                # Trailing stop: 2.0 * ATR above lowest close
                stoploss_price = lowest_close + 2.0 * atr[i]
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