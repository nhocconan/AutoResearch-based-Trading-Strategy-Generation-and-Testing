#!/usr/bin/env python3
"""
Experiment #406: 4h KAMA Trend + 1d/1w HMA Regime + RSI Pullback + ADX Filter

Hypothesis: After analyzing 405 failed experiments, the pattern is clear:
- Over-complicated strategies with too many filters = 0 trades or whipsaw
- Simple trend-following fails in 2022 crash and 2025 bear market
- Pure mean-reversion fails in strong trends

KEY INSIGHT: ASYMMETRIC REGIME-BASED TRADING
- In BULL regime (price > 1d KAMA): Only take LONG pullback entries
- In BEAR regime (price < 1d KAMA): Only take SHORT rally entries
- This avoids fighting the higher timeframe trend

STRATEGY COMPONENTS:
1. 1d KAMA(21) + 1w HMA(21): Dual HTF trend filter
   - KAMA adapts to volatility (better than EMA in chop)
   - 1w HMA provides ultra-smooth long-term bias
   - Both must agree for strong signal

2. 4h RSI(14) pullback entries:
   - Long: RSI 35-50 in bull regime (pullback, not oversold)
   - Short: RSI 50-65 in bear regime (rally, not overbought)
   - This catches continuations, not reversals

3. ADX(14) > 20: Minimum trend strength
   - Avoids entering in completely flat markets
   - Not too high (ADX>40 is rare)

4. ATR(14) trailing stop 2.5x: Risk management
   - Signal → 0 when price moves 2.5*ATR against position

5. Position sizing: 0.30 discrete
   - Conservative for 4h volatility
   - Discrete levels minimize fee churn

Why this should beat current best (Sharpe=0.676):
- Asymmetric logic avoids whipsaw in regime transitions
- KAMA adapts better than HMA/EMA in volatile periods
- RSI pullback (not extreme) generates more trades than RSI<30/>70
- Dual HTF (1d+1w) provides more stable trend bias than single HTF
- Should work on BTC/ETH/SOL individually (not SOL-biased)

Timeframe: 4h (REQUIRED)
HTF: 1d and 1w via mtf_data helper
Position sizing: 0.30 discrete levels
Stoploss: 2.5 * ATR(14) trailing
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_4h_kama_regime_1d_1w_rsi_pullback_adx_atr_v1"
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

def calculate_hma(close, period=21):
    """Calculate Hull Moving Average for smoother trend with less lag."""
    close_s = pd.Series(close)
    half = max(1, period // 2)
    sqrt_period = max(1, int(np.sqrt(period)))
    wma1 = close_s.ewm(span=half, min_periods=half, adjust=False).mean()
    wma2 = close_s.ewm(span=period, min_periods=period, adjust=False).mean()
    wma3 = (2 * wma1 - wma2).ewm(span=sqrt_period, min_periods=sqrt_period, adjust=False).mean()
    return wma3.values

def calculate_kama(close, er_period=10, fast_period=2, slow_period=30):
    """
    Calculate Kaufman Adaptive Moving Average.
    KAMA adapts to market volatility - moves fast in trends, slow in chop.
    Better than EMA for regime detection in volatile crypto markets.
    """
    n = len(close)
    kama = np.full(n, np.nan)
    
    # Calculate Efficiency Ratio (ER)
    er = np.full(n, np.nan)
    for i in range(er_period, n):
        signal = np.abs(close[i] - close[i - er_period])
        noise = np.sum(np.abs(np.diff(close[i - er_period:i + 1])))
        if noise > 1e-10:
            er[i] = signal / noise
        else:
            er[i] = 0.0
    
    # Calculate smoothing constant
    fast_sc = 2.0 / (fast_period + 1)
    slow_sc = 2.0 / (slow_period + 1)
    
    # Initialize first KAMA as SMA
    kama[er_period] = np.mean(close[:er_period + 1])
    
    # Calculate KAMA
    for i in range(er_period + 1, n):
        if not np.isnan(er[i]):
            sc = (er[i] * (fast_sc - slow_sc) + slow_sc) ** 2
            kama[i] = kama[i - 1] + sc * (close[i] - kama[i - 1])
    
    return kama

def calculate_rsi(close, period=14):
    """Calculate RSI using Wilder's smoothing method."""
    close_s = pd.Series(close)
    delta = close_s.diff()
    gain = delta.where(delta > 0, 0.0)
    loss = -delta.where(delta < 0, 0.0)
    avg_gain = gain.ewm(span=period, min_periods=period, adjust=False).mean()
    avg_loss = loss.ewm(span=period, min_periods=period, adjust=False).mean()
    rs = avg_gain / (avg_loss + 1e-10)
    rsi = 100 - (100 / (1 + rs))
    return rsi.values

def calculate_adx(high, low, close, period=14):
    """Calculate ADX for trend strength measurement."""
    n = len(close)
    adx = np.full(n, np.nan)
    
    # True Range
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]
    
    # Directional Movement
    dm_plus = np.zeros(n)
    dm_minus = np.zeros(n)
    
    for i in range(1, n):
        up_move = high[i] - high[i - 1]
        down_move = low[i - 1] - low[i]
        
        if up_move > down_move and up_move > 0:
            dm_plus[i] = up_move
        if down_move > up_move and down_move > 0:
            dm_minus[i] = down_move
    
    # Smooth with Wilder's method (EMA)
    tr_s = pd.Series(tr).ewm(span=period, min_periods=period, adjust=False).mean().values
    dm_plus_s = pd.Series(dm_plus).ewm(span=period, min_periods=period, adjust=False).mean().values
    dm_minus_s = pd.Series(dm_minus).ewm(span=period, min_periods=period, adjust=False).mean().values
    
    # DI+ and DI-
    di_plus = 100 * dm_plus_s / (tr_s + 1e-10)
    di_minus = 100 * dm_minus_s / (tr_s + 1e-10)
    
    # DX and ADX
    dx = 100 * np.abs(di_plus - di_minus) / (di_plus + di_minus + 1e-10)
    adx[period:] = pd.Series(dx).ewm(span=period, min_periods=period, adjust=False).mean().values[period:]
    
    return adx

def generate_signals(prices):
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    n = len(close)
    
    # Load HTF data ONCE before loop (Rule 1 - CRITICAL)
    df_1d = get_htf_data(prices, '1d')
    df_1w = get_htf_data(prices, '1w')
    
    # Calculate HTF indicators
    hma_1w = calculate_hma(df_1w['close'].values, 21)
    kama_1d = calculate_kama(df_1d['close'].values, er_period=10)
    
    # Align HTF to LTF (Rule 2 - auto shift(1) for completed bars)
    hma_1w_aligned = align_htf_to_ltf(prices, df_1w, hma_1w)
    kama_1d_aligned = align_htf_to_ltf(prices, df_1d, kama_1d)
    
    # Calculate 4h indicators
    atr = calculate_atr(high, low, close, 14)
    rsi = calculate_rsi(close, 14)
    adx = calculate_adx(high, low, close, 14)
    kama_4h = calculate_kama(close, er_period=10)
    
    signals = np.zeros(n)
    
    # Position sizing - discrete levels (Rule 4)
    SIZE = 0.30
    
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
        
        if np.isnan(kama_1d_aligned[i]) or np.isnan(hma_1w_aligned[i]):
            signals[i] = 0.0
            continue
        
        if np.isnan(rsi[i]) or np.isnan(adx[i]) or np.isnan(kama_4h[i]):
            signals[i] = 0.0
            continue
        
        # === REGIME DETECTION (Dual HTF) ===
        # Strong bull: price > 1d KAMA AND 1d KAMA > 1w HMA
        # Strong bear: price < 1d KAMA AND 1d KAMA < 1w HMA
        bull_regime = (close[i] > kama_1d_aligned[i]) and (kama_1d_aligned[i] > hma_1w_aligned[i])
        bear_regime = (close[i] < kama_1d_aligned[i]) and (kama_1d_aligned[i] < hma_1w_aligned[i])
        
        # === TREND STRENGTH FILTER ===
        trend_strong = adx[i] > 20.0
        
        # === RSI PULLBACK ENTRIES ===
        # Long: RSI 35-50 in bull regime (pullback, not oversold crash)
        rsi_pullback_long = 35.0 <= rsi[i] <= 50.0
        # Short: RSI 50-65 in bear regime (rally, not overbought spike)
        rsi_rally_short = 50.0 <= rsi[i] <= 65.0
        
        # === GENERATE SIGNAL ===
        new_signal = 0.0
        
        # BULL REGIME: Only long entries on pullback
        if bull_regime and trend_strong and rsi_pullback_long:
            new_signal = SIZE
        
        # BEAR REGIME: Only short entries on rally
        elif bear_regime and trend_strong and rsi_rally_short:
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
        # Exit long if regime turns bear or neutral
        if in_position and position_side > 0 and not bull_regime:
            new_signal = 0.0
        
        # Exit short if regime turns bull or neutral
        if in_position and position_side < 0 and not bear_regime:
            new_signal = 0.0
        
        # === UPDATE POSITION TRACKING ===
        if new_signal != 0.0:
            if not in_position:
                in_position = True
                position_side = np.sign(new_signal)
                highest_close = close[i] if position_side > 0 else 0.0
                lowest_close = close[i] if position_side < 0 else 0.0
            elif np.sign(new_signal) != position_side:
                # Position flip
                position_side = np.sign(new_signal)
                highest_close = close[i] if position_side > 0 else 0.0
                lowest_close = close[i] if position_side < 0 else 0.0
        else:
            if in_position:
                in_position = False
                position_side = 0
                highest_close = 0.0
                lowest_close = 0.0
        
        signals[i] = new_signal
    
    return signals