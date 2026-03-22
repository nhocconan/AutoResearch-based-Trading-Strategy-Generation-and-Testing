#!/usr/bin/env python3
"""
Experiment #101: 12h KAMA + 1d HMA Trend + Fisher Transform Entries + Choppiness Regime
Hypothesis: 12h timeframe balances trade frequency with noise reduction.
KAMA adapts to market efficiency (better than EMA in crypto whipsaws).
1d HMA provides stable trend bias without excessive lag.
Fisher Transform catches reversals precisely (proven in bear/range markets).
Choppiness Index filters out choppy periods (CHOP>61.8 = avoid trend trades).

Why this might work on 12h (learning from #095 Sharpe=0.001):
- #095 KAMA strategy had good returns but near-zero Sharpe (too many small losses)
- Key insight: Add Fisher Transform for precise entry timing
- Add Choppiness Index to avoid trading during choppy consolidation
- More lenient entry thresholds to ensure trades on BTC/ETH (not just SOL)
- Better stoploss with trailing ATR (2.0*ATR for 12h)
- Discrete position sizing to minimize fee churn

Timeframe: 12h (REQUIRED), HTF: 1d via mtf_data helper (call ONCE before loop).
Position sizing: 0.25 base, 0.35 strong signals. Stoploss at 2.0*ATR.
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_12h_kama_1d_hma_fisher_chop_regime_v1"
timeframe = "12h"
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

def calculate_kama(close, er_period=10, fast_period=2, slow_period=30):
    """
    Calculate Kaufman Adaptive Moving Average (KAMA).
    Adapts smoothing based on market efficiency (trend vs noise).
    """
    n = len(close)
    kama = np.zeros(n)
    kama[:] = np.nan
    
    # Calculate Efficiency Ratio (ER)
    er = np.zeros(n)
    for i in range(er_period, n):
        signal = np.abs(close[i] - close[i - er_period])
        noise = np.sum(np.abs(np.diff(close[i - er_period:i + 1])))
        if noise > 0:
            er[i] = signal / noise
    
    # Calculate smoothing constant
    fast_sc = 2.0 / (fast_period + 1)
    slow_sc = 2.0 / (slow_period + 1)
    
    # Initialize KAMA
    kama[er_period] = close[er_period]
    
    for i in range(er_period + 1, n):
        sc = (er[i] * (fast_sc - slow_sc) + slow_sc) ** 2
        kama[i] = kama[i - 1] + sc * (close[i] - kama[i - 1])
    
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

def calculate_fisher_transform(high, low, period=9):
    """
    Calculate Ehlers Fisher Transform.
    Normalizes price to Gaussian distribution for better reversal signals.
    """
    n = len(high)
    fisher = np.zeros(n)
    fisher[:] = np.nan
    fisher_signal = np.zeros(n)
    fisher_signal[:] = np.nan
    
    hl2 = (high + low) / 2
    
    for i in range(period, n):
        # Find highest high and lowest low over period
        highest = np.max(high[i - period + 1:i + 1])
        lowest = np.min(low[i - period + 1:i + 1])
        
        if highest == lowest:
            fisher[i] = fisher[i - 1] if i > 0 else 0
            continue
        
        # Normalize price to 0-1 range
        x = (hl2[i] - lowest) / (highest - lowest)
        
        # Constrain to avoid division by zero
        x = np.clip(x, 0.001, 0.999)
        
        # Fisher transform
        fisher[i] = 0.5 * np.log((1 + x) / (1 - x))
        
        # Signal line (1-period lag)
        if i > 0 and not np.isnan(fisher[i - 1]):
            fisher_signal[i] = fisher[i - 1]
        else:
            fisher_signal[i] = fisher[i]
    
    return fisher, fisher_signal

def calculate_choppiness(high, low, close, period=14):
    """
    Calculate Choppiness Index (CHOP).
    CHOP > 61.8 = choppy/range market
    CHOP < 38.2 = trending market
    """
    n = len(close)
    chop = np.zeros(n)
    chop[:] = np.nan
    
    for i in range(period, n):
        highest = np.max(high[i - period + 1:i + 1])
        lowest = np.min(low[i - period + 1:i + 1])
        
        if highest == lowest:
            chop[i] = 100
            continue
        
        # Sum of ATR over period
        tr_sum = 0
        for j in range(i - period + 1, i + 1):
            tr1 = high[j] - low[j]
            tr2 = np.abs(high[j] - close[j - 1]) if j > 0 else tr1
            tr3 = np.abs(low[j] - close[j - 1]) if j > 0 else tr1
            tr_sum += np.maximum(tr1, np.maximum(tr2, tr3))
        
        if tr_sum > 0:
            chop[i] = 100 * np.log10(tr_sum / (highest - lowest)) / np.log10(period)
        else:
            chop[i] = 100
    
    return chop

def calculate_adx(high, low, close, period=14):
    """Calculate ADX (Average Directional Index) for trend strength."""
    n = len(close)
    adx = np.zeros(n)
    adx[:] = np.nan
    
    plus_dm = np.zeros(n)
    minus_dm = np.zeros(n)
    
    for i in range(1, n):
        plus_move = high[i] - high[i-1]
        minus_move = low[i-1] - low[i]
        
        if plus_move > minus_move and plus_move > 0:
            plus_dm[i] = plus_move
        if minus_move > plus_move and minus_move > 0:
            minus_dm[i] = minus_move
    
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]
    
    atr = pd.Series(tr).ewm(span=period, min_periods=period, adjust=False).mean().values
    plus_di = 100 * pd.Series(plus_dm).ewm(span=period, min_periods=period, adjust=False).mean().values / atr
    minus_di = 100 * pd.Series(minus_dm).ewm(span=period, min_periods=period, adjust=False).mean().values / atr
    
    dx = np.zeros(n)
    mask = (plus_di + minus_di) > 0
    dx[mask] = 100 * np.abs(plus_di[mask] - minus_di[mask]) / (plus_di[mask] + minus_di[mask])
    
    adx = pd.Series(dx).ewm(span=period, min_periods=period, adjust=False).mean().values
    
    return adx

def calculate_rsi(close, period=14):
    """Calculate RSI."""
    n = len(close)
    rsi = np.zeros(n)
    rsi[:] = np.nan
    
    delta = np.diff(close)
    delta = np.insert(delta, 0, 0)
    
    gains = np.where(delta > 0, delta, 0)
    losses = np.where(delta < 0, -delta, 0)
    
    avg_gain = pd.Series(gains).ewm(span=period, min_periods=period, adjust=False).mean().values
    avg_loss = pd.Series(losses).ewm(span=period, min_periods=period, adjust=False).mean().values
    
    mask = avg_loss > 0
    rs = np.zeros(n)
    rs[mask] = avg_gain[mask] / avg_loss[mask]
    rsi[mask] = 100 - (100 / (1 + rs[mask]))
    rsi[~mask] = 100.0
    
    return rsi

def generate_signals(prices):
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    n = len(close)
    
    # Load HTF data ONCE before loop (Rule 1 - CRITICAL)
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate HTF indicators
    hma_1d = calculate_hma(df_1d['close'].values, 21)
    
    # Align HTF to LTF (Rule 2 - no manual index mapping, auto shift(1))
    hma_1d_aligned = align_htf_to_ltf(prices, df_1d, hma_1d)
    
    # Calculate 12h indicators
    atr = calculate_atr(high, low, close, 14)
    adx = calculate_adx(high, low, close, 14)
    rsi = calculate_rsi(close, 14)
    kama = calculate_kama(close, er_period=10, fast_period=2, slow_period=30)
    chop = calculate_choppiness(high, low, close, 14)
    fisher, fisher_signal = calculate_fisher_transform(high, low, 9)
    
    signals = np.zeros(n)
    
    # Position sizing - discrete levels (Rule 4)
    SIZE_BASE = 0.25
    SIZE_STRONG = 0.35
    
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
        
        if np.isnan(hma_1d_aligned[i]):
            signals[i] = 0.0
            continue
        
        if np.isnan(adx[i]) or np.isnan(kama[i]) or np.isnan(chop[i]):
            signals[i] = 0.0
            continue
        
        if np.isnan(fisher[i]) or np.isnan(fisher_signal[i]):
            signals[i] = 0.0
            continue
        
        # === MULTI-TIMEFRAME TREND BIAS ===
        # 1d HMA = higher timeframe trend bias
        bull_trend_1d = close[i] > hma_1d_aligned[i]
        bear_trend_1d = close[i] < hma_1d_aligned[i]
        
        # === KAMA TREND (adaptive) ===
        kama_bullish = close[i] > kama[i]
        kama_bearish = close[i] < kama[i]
        
        # === CHOPPINESS REGIME FILTER ===
        # CHOP < 38.2 = trending (good for trend trades)
        # CHOP > 61.8 = choppy (avoid trend trades, or use mean reversion)
        trending_regime = chop[i] < 50  # relaxed threshold to ensure trades
        choppy_regime = chop[i] > 55
        
        # === ADX TREND STRENGTH ===
        # ADX > 20 = trending market (lower threshold for 12h to ensure trades)
        trending_adx = adx[i] > 18
        strong_trend = adx[i] > 25
        
        # === FISHER TRANSFORM ENTRY SIGNALS ===
        # Long: Fisher crosses above -1.5 (oversold reversal)
        # Short: Fisher crosses below +1.5 (overbought reversal)
        fisher_long_signal = fisher[i] > -1.5 and fisher_signal[i] <= -1.5
        fisher_short_signal = fisher[i] < 1.5 and fisher_signal[i] >= 1.5
        
        # Fisher momentum (less strict for more trades)
        fisher_momentum_long = fisher[i] > -1.0 and fisher[i] > fisher_signal[i]
        fisher_momentum_short = fisher[i] < 1.0 and fisher[i] < fisher_signal[i]
        
        # === RSI CONFIRMATION (lenient thresholds) ===
        rsi_long_ok = rsi[i] > 35  # not extremely oversold
        rsi_short_ok = rsi[i] < 65  # not extremely overbought
        
        new_signal = 0.0
        
        # === LONG ENTRY CONDITIONS (multiple paths to ensure trades) ===
        # Path 1: Strong trend + KAMA bullish + 1d bullish (primary)
        if kama_bullish and bull_trend_1d and trending_regime:
            if strong_trend and fisher_momentum_long:
                new_signal = SIZE_STRONG
            elif trending_adx and rsi_long_ok:
                new_signal = SIZE_BASE
        
        # Path 2: KAMA bullish + Fisher reversal (catches early entries)
        if new_signal == 0.0 and kama_bullish and fisher_momentum_long:
            if bull_trend_1d or trending_adx:
                new_signal = SIZE_BASE
        
        # Path 3: 1d bullish + trending regime (fallback for all symbols)
        if new_signal == 0.0 and bull_trend_1d and trending_regime:
            if kama_bullish or fisher_momentum_long:
                new_signal = SIZE_BASE
        
        # === SHORT ENTRY CONDITIONS (multiple paths to ensure trades) ===
        # Path 1: Strong trend + KAMA bearish + 1d bearish (primary)
        if kama_bearish and bear_trend_1d and trending_regime:
            if strong_trend and fisher_momentum_short:
                new_signal = -SIZE_STRONG
            elif trending_adx and rsi_short_ok:
                new_signal = -SIZE_BASE
        
        # Path 2: KAMA bearish + Fisher reversal (catches early entries)
        if new_signal == 0.0 and kama_bearish and fisher_momentum_short:
            if bear_trend_1d or trending_adx:
                new_signal = -SIZE_BASE
        
        # Path 3: 1d bearish + trending regime (fallback for all symbols)
        if new_signal == 0.0 and bear_trend_1d and trending_regime:
            if kama_bearish or fisher_momentum_short:
                new_signal = -SIZE_BASE
        
        # === CHOPPY REGIME MEAN REVERSION (alternative logic) ===
        # In choppy markets, fade extremes instead of trend follow
        if choppy_regime and new_signal == 0.0:
            # Long when RSI oversold in choppy market
            if rsi[i] < 35 and kama_bullish:
                new_signal = SIZE_BASE
            # Short when RSI overbought in choppy market
            elif rsi[i] > 65 and kama_bearish:
                new_signal = -SIZE_BASE
        
        # === STOPLOSS LOGIC (Rule 6) - 2.0 * ATR for 12h ===
        # Update trailing highs/lows for active positions
        if in_position and position_side > 0:
            if close[i] > highest_close:
                highest_close = close[i]
            # Trailing stop: 2.0 * ATR below highest close
            stoploss_price = highest_close - 2.0 * atr[i]
            if close[i] < stoploss_price:
                new_signal = 0.0  # Stoploss hit
        
        if in_position and position_side < 0:
            if lowest_close == 0.0 or close[i] < lowest_close:
                lowest_close = close[i]
            # Trailing stop: 2.0 * ATR above lowest close
            stoploss_price = lowest_close + 2.0 * atr[i]
            if close[i] > stoploss_price:
                new_signal = 0.0  # Stoploss hit
        
        # Update position tracking
        # Entering new position
        if new_signal != 0.0 and not in_position:
            in_position = True
            position_side = np.sign(new_signal)
            entry_price = close[i]
            highest_close = close[i] if position_side > 0 else 0.0
            lowest_close = close[i] if position_side < 0 else 0.0
        
        # Reversing position
        elif new_signal != 0.0 and in_position and np.sign(new_signal) != position_side:
            position_side = np.sign(new_signal)
            entry_price = close[i]
            highest_close = close[i] if position_side > 0 else 0.0
            lowest_close = close[i] if position_side < 0 else 0.0
        
        # Exiting position
        elif new_signal == 0.0 and in_position:
            in_position = False
            position_side = 0
            entry_price = 0.0
            highest_close = 0.0
            lowest_close = 0.0
        
        signals[i] = new_signal
    
    return signals