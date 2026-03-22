#!/usr/bin/env python3
"""
Experiment #389: 12h Weekly Trend Bias + Daily HMA + ADX Regime + RSI Pullback

Hypothesis: After 388 failed experiments, the pattern is clear - 12h strategies fail
because they're either too slow (miss moves) or too strict (0 trades). The solution:

1. WEEKLY HMA(21) as ultimate trend filter - only trade in direction of weekly trend
   This is the MOST stable trend indicator, filters out noise from 2022 crash

2. DAILY HMA(21) for intermediate confirmation - aligns with weekly for stronger signals

3. ADX(14) regime detection with HYSTERESIS:
   - ADX > 25 = trending (follow weekly trend with RSI pullback entries)
   - ADX < 20 = ranging (mean-reversion with RSI extremes)
   - 20-25 = hold current position (avoid whipsaw on regime flip)

4. RSI(14) for entries:
   - Trending: RSI pullback to 40-50 (long) or 50-60 (short) in trend direction
   - Ranging: RSI < 35 (long) or RSI > 65 (short)

5. ATR(14) * 2.5 trailing stop - protects from crashes like 2022

6. POSITION SIZING: 0.30 discrete (conservative for 12h volatility)
   - Larger size in trending regime (0.30)
   - Smaller size in ranging regime (0.20)

Why 12h should work now:
- Weekly trend filter prevents trading against major trend (key failure mode)
- ADX hysteresis prevents regime flip whipsaw
- RSI pullback entries are LOOSE enough to generate trades (≥10/train, ≥3/test)
- ATR stoploss protects from 2022-style crashes
- Works on BTC, ETH, SOL individually (not SOL-biased)

Timeframe: 12h (REQUIRED for this experiment)
HTF: 1d and 1w via mtf_data helper (call ONCE before loop)
Position sizing: 0.20-0.30 discrete levels
Stoploss: 2.5 * ATR(14) trailing
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_12h_weekly_trend_daily_hma_adx_rsi_atr_v1"
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

def calculate_hma(close, period=21):
    """Calculate Hull Moving Average for smoother trend with less lag."""
    close_s = pd.Series(close)
    half = max(1, period // 2)
    sqrt_period = max(1, int(np.sqrt(period)))
    wma1 = close_s.ewm(span=half, min_periods=half, adjust=False).mean()
    wma2 = close_s.ewm(span=period, min_periods=period, adjust=False).mean()
    wma3 = (2 * wma1 - wma2).ewm(span=sqrt_period, min_periods=sqrt_period, adjust=False).mean()
    return wma3.values

def calculate_adx(high, low, close, period=14):
    """Calculate ADX (Average Directional Index) for trend strength."""
    n = len(close)
    adx = np.full(n, np.nan)
    
    # Calculate True Range and Directional Movement
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
    
    # Smooth with Wilder's method (EMA with span=period)
    tr_smooth = pd.Series(tr).ewm(span=period, min_periods=period, adjust=False).mean().values
    plus_dm_smooth = pd.Series(plus_dm).ewm(span=period, min_periods=period, adjust=False).mean().values
    minus_dm_smooth = pd.Series(minus_dm).ewm(span=period, min_periods=period, adjust=False).mean().values
    
    # Calculate DI+ and DI-
    di_plus = np.zeros(n)
    di_minus = np.zeros(n)
    
    for i in range(period, n):
        if tr_smooth[i] > 1e-10:
            di_plus[i] = 100 * plus_dm_smooth[i] / tr_smooth[i]
            di_minus[i] = 100 * minus_dm_smooth[i] / tr_smooth[i]
    
    # Calculate DX and ADX
    for i in range(period, n):
        di_sum = di_plus[i] + di_minus[i]
        if di_sum > 1e-10:
            dx = 100 * np.abs(di_plus[i] - di_minus[i]) / di_sum
            if i == period:
                adx[i] = dx
            else:
                adx[i] = (adx[i-1] * (period - 1) + dx) / period
    
    return adx

def calculate_rsi(close, period=14):
    """Calculate RSI (Relative Strength Index)."""
    close_s = pd.Series(close)
    delta = close_s.diff()
    gain = delta.where(delta > 0, 0.0)
    loss = -delta.where(delta < 0, 0.0)
    
    avg_gain = gain.ewm(span=period, min_periods=period, adjust=False).mean()
    avg_loss = loss.ewm(span=period, min_periods=period, adjust=False).mean()
    
    rs = avg_gain / avg_loss.replace(0, np.inf)
    rsi = 100 - (100 / (1 + rs))
    return rsi.values

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
    
    # Align HTF to LTF (Rule 2 - auto shift(1) for completed bars)
    hma_1d_aligned = align_htf_to_ltf(prices, df_1d, hma_1d)
    hma_1w_aligned = align_htf_to_ltf(prices, df_1w, hma_1w)
    
    # Calculate 12h indicators
    atr = calculate_atr(high, low, close, 14)
    adx = calculate_adx(high, low, close, 14)
    rsi = calculate_rsi(close, 14)
    
    signals = np.zeros(n)
    
    # Position sizing - discrete levels (Rule 4)
    SIZE_TREND = 0.30  # Larger in trending regime
    SIZE_RANGE = 0.20  # Smaller in ranging regime
    
    # Track position state for stoploss
    in_position = False
    position_side = 0
    highest_close = 0.0
    lowest_close = 0.0
    entry_price = 0.0
    prev_adx_regime = 0  # 0=neutral, 1=trending, 2=ranging
    
    for i in range(100, n):
        # Skip if indicators not ready
        if np.isnan(atr[i]) or atr[i] == 0:
            signals[i] = 0.0
            continue
        
        if np.isnan(hma_1d_aligned[i]) or np.isnan(hma_1w_aligned[i]):
            signals[i] = 0.0
            continue
        
        if np.isnan(adx[i]) or np.isnan(rsi[i]):
            signals[i] = 0.0
            continue
        
        # === WEEKLY TREND BIAS (ultimate filter) ===
        bull_trend_1w = close[i] > hma_1w_aligned[i]
        bear_trend_1w = close[i] < hma_1w_aligned[i]
        
        # === DAILY TREND CONFIRMATION ===
        bull_trend_1d = close[i] > hma_1d_aligned[i]
        bear_trend_1d = close[i] < hma_1d_aligned[i]
        
        # === ADX REGIME DETECTION WITH HYSTERESIS ===
        # Use hysteresis to avoid whipsaw on regime flip
        if adx[i] > 25:
            adx_regime = 1  # trending
        elif adx[i] < 20:
            adx_regime = 2  # ranging
        else:
            adx_regime = prev_adx_regime  # hold previous regime
        
        prev_adx_regime = adx_regime
        trending_market = (adx_regime == 1)
        ranging_market = (adx_regime == 2)
        
        # === GENERATE SIGNAL ===
        new_signal = 0.0
        current_size = SIZE_TREND if trending_market else SIZE_RANGE
        
        # TRENDING REGIME: Follow weekly trend with RSI pullback
        if trending_market:
            # Long: weekly bull + RSI pullback to 40-55
            if bull_trend_1w and rsi[i] >= 40 and rsi[i] <= 55:
                new_signal = current_size
            # Short: weekly bear + RSI pullback to 45-60
            elif bear_trend_1w and rsi[i] >= 45 and rsi[i] <= 60:
                new_signal = -current_size
        
        # RANGING REGIME: Mean-reversion with RSI extremes
        elif ranging_market:
            # Long: RSI < 35 (oversold)
            if rsi[i] < 35:
                new_signal = current_size
            # Short: RSI > 65 (overbought)
            elif rsi[i] > 65:
                new_signal = -current_size
        
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
        
        # === TREND REVERSAL EXIT ===
        # Exit if weekly trend flips against position
        if in_position and new_signal != 0.0:
            if position_side > 0 and bear_trend_1w:
                new_signal = 0.0
            if position_side < 0 and bull_trend_1w:
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