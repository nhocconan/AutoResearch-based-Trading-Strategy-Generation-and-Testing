#!/usr/bin/env python3
"""
Experiment #350: 30m Regime-Adaptive Strategy with 4h KAMA Bias + ADX + RSI + Volume

Hypothesis: After 298 failed strategies, the key insight is that 30m needs a balance
between noise filtering and signal generation. This strategy uses:

1. KAUFMAN ADAPTIVE MA (KAMA) instead of HMA/EMA:
   - KAMA adapts smoothing based on market efficiency (volatility)
   - Less lag in trends, more smoothing in ranges
   - Proven to work better on crypto's varying volatility regimes

2. ADX for regime detection (more reliable than Choppiness):
   - ADX > 25 = trending → follow 4h KAMA direction
   - ADX < 20 = ranging → mean reversion with RSI extremes
   - 20 <= ADX <= 25 = transition → reduce position or flat

3. 4h KAMA for directional bias:
   - Only long if price > 4h KAMA (bullish bias)
   - Only short if price < 4h KAMA (bearish bias)
   - More stable than 4h HMA for crypto's noise

4. RSI(14) with LOOSE thresholds for entries:
   - Long: RSI < 40 (not 30) + bullish conditions
   - Short: RSI > 60 (not 70) + bearish conditions
   - Looser = more trades (critical for meeting minimum trade count)

5. Volume confirmation:
   - Volume > 0.8 * MA(Volume, 20) to avoid low-liquidity traps
   - Reduces false breakouts

6. Position sizing: 0.25 discrete, ATR stoploss at 2.0x

Why 30m specifically:
- Fast enough to catch intraday moves
- Slow enough to avoid 5m/15m noise
- 4h HTF gives 8 bars per 4h candle (good MTF ratio)

Timeframe: 30m (REQUIRED for this experiment)
HTF: 4h via mtf_data helper (call ONCE before loop)
Position sizing: 0.25 discrete levels
Stoploss: 2.0 * ATR(14) trailing
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_30m_regime_adx_4h_kama_rsi_vol_atr_v1"
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

def calculate_kama(close, er_period=10, fast_period=2, slow_period=30):
    """
    Calculate Kaufman Adaptive Moving Average (KAMA).
    KAMA adapts smoothing based on market efficiency ratio.
    
    ER = |Close - Close[n]| / Sum(|Close[i] - Close[i-1]|)
    SC = (ER * (fast_sc - slow_sc) + slow_sc)^2
    KAMA = KAMA_prev + SC * (Close - KAMA_prev)
    """
    n = len(close)
    kama = np.full(n, np.nan)
    
    if n < er_period + slow_period:
        return kama
    
    # Calculate Efficiency Ratio (ER)
    er = np.zeros(n)
    for i in range(er_period, n):
        signal = np.abs(close[i] - close[i - er_period])
        noise = np.sum(np.abs(np.diff(close[i - er_period:i + 1])))
        if noise > 1e-10:
            er[i] = signal / noise
        else:
            er[i] = 0.0
    
    # Calculate Smoothing Constant (SC)
    fast_sc = 2.0 / (fast_period + 1)
    slow_sc = 2.0 / (slow_period + 1)
    sc = (er * (fast_sc - slow_sc) + slow_sc) ** 2
    
    # Initialize KAMA
    kama[er_period] = close[er_period]
    
    # Calculate KAMA
    for i in range(er_period + 1, n):
        kama[i] = kama[i - 1] + sc[i] * (close[i] - kama[i - 1])
    
    return kama

def calculate_rsi(close, period=14):
    """Calculate RSI indicator."""
    close_s = pd.Series(close)
    delta = close_s.diff()
    gain = delta.where(delta > 0, 0.0)
    loss = -delta.where(delta < 0, 0.0)
    avg_gain = gain.ewm(span=period, min_periods=period, adjust=False).mean()
    avg_loss = loss.ewm(span=period, min_periods=period, adjust=False).mean()
    rs = avg_gain / np.maximum(avg_loss, 1e-10)
    rsi = 100 - (100 / (1 + rs))
    return rsi.values

def calculate_adx(high, low, close, period=14):
    """
    Calculate Average Directional Index (ADX).
    ADX > 25 = trending, ADX < 20 = ranging
    """
    n = len(close)
    adx = np.full(n, np.nan)
    
    if n < period * 2:
        return adx
    
    # Calculate True Range and Directional Movement
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]
    
    plus_dm = np.zeros(n)
    minus_dm = np.zeros(n)
    
    for i in range(1, n):
        plus_move = high[i] - high[i - 1]
        minus_move = low[i - 1] - low[i]
        
        if plus_move > minus_move and plus_move > 0:
            plus_dm[i] = plus_move
        if minus_move > plus_move and minus_move > 0:
            minus_dm[i] = minus_move
    
    # Smooth with Wilder's method (EMA with span=period)
    tr_smooth = pd.Series(tr).ewm(span=period, min_periods=period, adjust=False).mean().values
    plus_dm_smooth = pd.Series(plus_dm).ewm(span=period, min_periods=period, adjust=False).mean().values
    minus_dm_smooth = pd.Series(minus_dm).ewm(span=period, min_periods=period, adjust=False).mean().values
    
    # Calculate DI+ and DI-
    plus_di = 100 * plus_dm_smooth / np.maximum(tr_smooth, 1e-10)
    minus_di = 100 * minus_dm_smooth / np.maximum(tr_smooth, 1e-10)
    
    # Calculate DX and ADX
    dx = 100 * np.abs(plus_di - minus_di) / np.maximum(plus_di + minus_di, 1e-10)
    adx_series = pd.Series(dx).ewm(span=period, min_periods=period, adjust=False).mean()
    adx = adx_series.values
    
    return adx

def calculate_volume_ma(volume, period=20):
    """Calculate moving average of volume."""
    vol_s = pd.Series(volume)
    vol_ma = vol_s.rolling(window=period, min_periods=period).mean().values
    return vol_ma

def generate_signals(prices):
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    volume = prices["volume"].values
    n = len(close)
    
    # Load HTF data ONCE before loop (Rule 1 - CRITICAL)
    df_4h = get_htf_data(prices, '4h')
    
    # Calculate HTF indicators
    kama_4h = calculate_kama(df_4h['close'].values, er_period=10, fast_period=2, slow_period=30)
    
    # Align HTF to LTF (Rule 2 - auto shift(1) for completed bars)
    kama_4h_aligned = align_htf_to_ltf(prices, df_4h, kama_4h)
    
    # Calculate 30m indicators
    atr = calculate_atr(high, low, close, 14)
    kama_30m = calculate_kama(close, er_period=10, fast_period=2, slow_period=30)
    rsi = calculate_rsi(close, 14)
    adx = calculate_adx(high, low, close, 14)
    vol_ma = calculate_volume_ma(volume, 20)
    
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
        
        if np.isnan(kama_4h_aligned[i]):
            signals[i] = 0.0
            continue
        
        if np.isnan(kama_30m[i]) or np.isnan(rsi[i]) or np.isnan(adx[i]):
            signals[i] = 0.0
            continue
        
        if np.isnan(vol_ma[i]) or vol_ma[i] == 0:
            signals[i] = 0.0
            continue
        
        # === REGIME DETECTION (ADX) ===
        in_trend = adx[i] > 25  # Trending market
        in_range = adx[i] < 20  # Ranging market
        # 20 <= adx <= 25 = transition
        
        # === 4h KAMA TREND BIAS ===
        bull_trend_4h = close[i] > kama_4h_aligned[i]
        bear_trend_4h = close[i] < kama_4h_aligned[i]
        
        # === 30m KAMA DIRECTION ===
        kama_bull_30m = close[i] > kama_30m[i]
        kama_bear_30m = close[i] < kama_30m[i]
        
        # === RSI SIGNALS (LOOSE THRESHOLDS for more trades) ===
        rsi_oversold = rsi[i] < 40  # Loosened from 30
        rsi_overbought = rsi[i] > 60  # Loosened from 70
        
        # === VOLUME CONFIRMATION ===
        vol_confirmed = volume[i] > 0.8 * vol_ma[i]
        
        # === GENERATE SIGNAL BASED ON REGIME ===
        new_signal = 0.0
        
        # TREND REGIME: Follow 4h KAMA direction with 30m confirmation
        if in_trend:
            # Long: 4h bullish + 30m bullish + RSI not overbought + volume
            if bull_trend_4h and kama_bull_30m and not rsi_overbought and vol_confirmed:
                new_signal = SIZE
            
            # Short: 4h bearish + 30m bearish + RSI not oversold + volume
            elif bear_trend_4h and kama_bear_30m and not rsi_oversold and vol_confirmed:
                new_signal = -SIZE
        
        # RANGE REGIME: Mean reversion with 4h bias
        elif in_range:
            # Long: RSI oversold + price above 4h KAMA (bullish bias)
            if rsi_oversold and bull_trend_4h:
                new_signal = SIZE
            
            # Short: RSI overbought + price below 4h KAMA (bearish bias)
            elif rsi_overbought and bear_trend_4h:
                new_signal = -SIZE
        
        # TRANSITION REGIME: Maintain existing position, no new entries
        if not in_trend and not in_range:
            if in_position:
                new_signal = signals[i - 1] if i > 0 else 0.0
            else:
                new_signal = 0.0
        
        # === STOPLOSS LOGIC (Rule 6) - 2.0 * ATR trailing ===
        if in_position and position_side != 0:
            if position_side > 0:
                if close[i] > highest_close:
                    highest_close = close[i]
                stoploss_price = highest_close - 2.0 * atr[i]
                if close[i] < stoploss_price:
                    new_signal = 0.0
            
            if position_side < 0:
                if lowest_close == 0.0 or close[i] < lowest_close:
                    lowest_close = close[i]
                stoploss_price = lowest_close + 2.0 * atr[i]
                if close[i] > stoploss_price:
                    new_signal = 0.0
        
        # === TREND REVERSAL EXIT ===
        # Exit if 4h trend flips against position
        if in_position and new_signal != 0.0:
            if position_side > 0 and bear_trend_4h:
                new_signal = 0.0
            if position_side < 0 and bull_trend_4h:
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