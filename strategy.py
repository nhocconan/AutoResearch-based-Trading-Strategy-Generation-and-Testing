#!/usr/bin/env python3
"""
Experiment #436: 4h Regime-Adaptive Multi-Signal with Daily Trend Filter

Hypothesis: After 435 experiments, the key insight is that 4h strategies fail because
they're either too strict (0 trades) or don't adapt to market regime. This strategy:

1. DAILY HMA(21) TREND BIAS (via mtf_data helper):
   - Long bias when price > 1d HMA (bull regime)
   - Short bias when price < 1d HMA (bear regime)
   - HMA smoother than EMA, critical for daily trend

2. FOUR SIGNAL TYPES (any can trigger - loose thresholds for trade frequency):
   a) RSI(14) MEAN REVERSION: RSI < 35 (long) or > 65 (short)
      - Looser than 30/70 to ensure >10 trades/year
      - Must align with daily trend bias
   
   b) VOL SPIKE CONTRARIAN: ATR(7)/ATR(30) > 1.5
      - Lower threshold than failed experiments (was 1.8-2.0)
      - Captures panic reversals without waiting for extreme
   
   c) BB MEAN REVERSION: Price < BB_lower (long) or > BB_upper (short)
      - Works in ranging markets (ADX < 20)
      - 2.0 std dev bands
   
   d) MOMENTUM CONTINUATION: ROC(10) > 5% + ADX > 20 (long)
      - Catches trending moves in direction of daily bias
      - Short when ROC < -5% + ADX > 20

3. ADX(14) REGIME FILTER:
   - ADX > 20 = trending (allow momentum + breakout)
   - ADX < 20 = ranging (allow mean reversion only)
   - Lower threshold than failed experiments (was 22-25)

4. ATR(14) TRAILING STOP at 2.5x:
   - Signal → 0 when price moves 2.5*ATR against position
   - Critical for crash protection

5. POSITION SIZING: 0.25 discrete (conservative for 4h volatility)
   - Max 25% capital per position
   - Discrete levels: 0.0, ±0.25

Why this should beat Sharpe=0.676:
- Looser thresholds ensure sufficient trades (>10/year per symbol)
- Multiple signal types = more entry opportunities
- Daily HMA filter prevents counter-trend disasters
- Vol spike at 1.5 ratio catches more reversals than 1.8+
- Should work on BTC/ETH/SOL individually (not SOL-biased)

Timeframe: 4h (REQUIRED for this experiment)
HTF: 1d via mtf_data helper (call ONCE before loop)
Position sizing: 0.25 discrete levels
Stoploss: 2.5 * ATR(14) trailing
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_4h_regime_adaptive_daily_hma_multi_signal_atr_v1"
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

def calculate_adx(high, low, close, period=14):
    """Calculate Average Directional Index (ADX)."""
    n = len(close)
    adx = np.full(n, np.nan)
    
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]
    
    plus_dm = np.zeros(n)
    minus_dm = np.zeros(n)
    
    for i in range(1, n):
        plus_move = high[i] - high[i-1]
        minus_move = low[i-1] - low[i]
        
        if plus_move > minus_move and plus_move > 0:
            plus_dm[i] = plus_move
        if minus_move > plus_move and minus_move > 0:
            minus_dm[i] = minus_move
    
    plus_dm_s = pd.Series(plus_dm).ewm(span=period, min_periods=period, adjust=False).mean().values
    minus_dm_s = pd.Series(minus_dm).ewm(span=period, min_periods=period, adjust=False).mean().values
    tr_s = pd.Series(tr).ewm(span=period, min_periods=period, adjust=False).mean().values
    
    for i in range(period, n):
        if tr_s[i] > 1e-10:
            plus_di = 100 * plus_dm_s[i] / tr_s[i]
            minus_di = 100 * minus_dm_s[i] / tr_s[i]
            di_sum = plus_di + minus_di
            if di_sum > 1e-10:
                dx = 100 * np.abs(plus_di - minus_di) / di_sum
            else:
                dx = 0
        else:
            dx = 0
        
        if i == period:
            adx[i] = dx
        else:
            adx[i] = ((adx[i-1] * (period - 1)) + dx) / period
    
    return adx

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

def calculate_bollinger_bands(close, period=20, std_dev=2.0):
    """Calculate Bollinger Bands."""
    close_s = pd.Series(close)
    sma = close_s.rolling(window=period, min_periods=period).mean()
    std = close_s.rolling(window=period, min_periods=period).std()
    upper = sma + (std_dev * std)
    lower = sma - (std_dev * std)
    return upper.values, lower.values

def calculate_roc(close, period=10):
    """Calculate Rate of Change."""
    close_s = pd.Series(close)
    roc = close_s.pct_change(periods=period) * 100
    return roc.values

def generate_signals(prices):
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    n = len(close)
    
    # Load HTF data ONCE before loop (Rule 1 - CRITICAL)
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate HTF indicators
    hma_1d = calculate_hma(df_1d['close'].values, 21)
    
    # Align HTF to LTF (Rule 2 - auto shift(1) for completed bars)
    hma_1d_aligned = align_htf_to_ltf(prices, df_1d, hma_1d)
    
    # Calculate 4h indicators
    atr = calculate_atr(high, low, close, 14)
    atr_7 = calculate_atr(high, low, close, 7)
    atr_30 = calculate_atr(high, low, close, 30)
    adx = calculate_adx(high, low, close, 14)
    rsi = calculate_rsi(close, 14)
    bb_upper, bb_lower = calculate_bollinger_bands(close, 20, 2.0)
    roc = calculate_roc(close, 10)
    
    signals = np.zeros(n)
    
    # Position sizing - discrete levels (Rule 4)
    SIZE = 0.25
    
    # Track position state for stoploss
    in_position = False
    position_side = 0
    highest_close = 0.0
    lowest_close = 0.0
    entry_price = 0.0
    
    for i in range(100, n):
        # Skip if indicators not ready
        if np.isnan(atr[i]) or atr[i] == 0:
            signals[i] = 0.0
            continue
        
        if np.isnan(hma_1d_aligned[i]):
            signals[i] = 0.0
            continue
        
        if np.isnan(adx[i]) or np.isnan(rsi[i]):
            signals[i] = 0.0
            continue
        
        if np.isnan(bb_upper[i]) or np.isnan(bb_lower[i]):
            signals[i] = 0.0
            continue
        
        if np.isnan(atr_7[i]) or np.isnan(atr_30[i]):
            signals[i] = 0.0
            continue
        
        if np.isnan(roc[i]):
            signals[i] = 0.0
            continue
        
        # === DAILY HMA TREND BIAS ===
        bull_trend_1d = close[i] > hma_1d_aligned[i]
        bear_trend_1d = close[i] < hma_1d_aligned[i]
        
        # === REGIME DETECTION ===
        trending_market = adx[i] > 20  # Lower threshold for more trades
        ranging_market = adx[i] <= 20
        
        # === SIGNAL 1: RSI MEAN REVERSION ===
        rsi_long = rsi[i] < 35  # Looser than 30
        rsi_short = rsi[i] > 65  # Looser than 70
        
        # === SIGNAL 2: VOL SPIKE CONTRARIAN ===
        vol_ratio = atr_7[i] / atr_30[i] if atr_30[i] > 0 else 0
        vol_spike = vol_ratio > 1.5  # Lower than failed 1.8-2.0
        vol_long = vol_spike and close[i] < bb_lower[i]  # Panic at BB lower
        vol_short = vol_spike and close[i] > bb_upper[i]  # Spike at BB upper
        
        # === SIGNAL 3: BB MEAN REVERSION (ranging markets) ===
        bb_long = close[i] < bb_lower[i]
        bb_short = close[i] > bb_upper[i]
        
        # === SIGNAL 4: MOMENTUM CONTINUATION (trending markets) ===
        momentum_long = roc[i] > 3.0  # Lower threshold
        momentum_short = roc[i] < -3.0  # Lower threshold
        
        # === GENERATE SIGNAL (Ensemble - any signal can trigger) ===
        new_signal = 0.0
        
        # RSI MEAN REVERSION (works in any regime, must align with daily trend)
        if rsi_long and bull_trend_1d:
            new_signal = SIZE
        elif rsi_short and bear_trend_1d:
            new_signal = -SIZE
        
        # VOL SPIKE CONTRARIAN (works in any regime, must align with daily trend)
        if new_signal == 0.0:
            if vol_long and bull_trend_1d:
                new_signal = SIZE
            elif vol_short and bear_trend_1d:
                new_signal = -SIZE
        
        # BB MEAN REVERSION (only in ranging market)
        if new_signal == 0.0 and ranging_market:
            if bb_long and bull_trend_1d:
                new_signal = SIZE
            elif bb_short and bear_trend_1d:
                new_signal = -SIZE
        
        # MOMENTUM CONTINUATION (only in trending market)
        if new_signal == 0.0 and trending_market:
            if momentum_long and bull_trend_1d:
                new_signal = SIZE
            elif momentum_short and bear_trend_1d:
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
        
        # === TREND REVERSAL EXIT ===
        if in_position and new_signal != 0.0:
            if position_side > 0 and bear_trend_1d:
                new_signal = 0.0
            if position_side < 0 and bull_trend_1d:
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