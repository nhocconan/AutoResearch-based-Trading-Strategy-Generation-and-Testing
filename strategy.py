#!/usr/bin/env python3
"""
Experiment #452: 30m Regime-Adaptive Multi-Signal with 4h/1d Trend Filter

Hypothesis: After 451 failed experiments, the key insight is that FIXED strategies
fail because markets regime-shift. This strategy ADAPTS to market conditions:

1. 1D ADX REGIME FILTER (via mtf_data):
   - ADX > 25 = trending (use trend-following signals)
   - ADX < 20 = ranging (use mean-reversion signals)
   - 20-25 = transition (reduce position size by 50%)

2. 4H HMA(21) TREND BIAS (via mtf_data):
   - Long bias when price > 4h HMA
   - Short bias when price < 4h HMA
   - HMA smoother than EMA, critical for trend detection

3. 30M SIGNAL TYPES (regime-dependent):
   a) TRENDING (ADX>25): KAMA(21) crossover + pullback to EMA(8)
      - KAMA adapts to volatility, less whipsaw than EMA
      - Enter on pullback in direction of 4h trend
   
   b) RANGING (ADX<20): RSI(14) extremes + Bollinger Band touch
      - Long: RSI<35 + price<BBLow
      - Short: RSI>65 + price>BBHigh
      - Must align with 4h HMA bias

4. VOLUME CONFIRMATION:
   - Volume ratio (vol/MA20) > 1.5 for breakouts
   - Prevents false signals on low-volume moves

5. ATR(14) TRAILING STOP at 2.5x:
   - Signal → 0 when price moves 2.5*ATR against position
   - Critical for crash protection

6. POSITION SIZING: 0.30 discrete (0.15 in transition regime)
   - Max 30% capital per position
   - Discrete levels minimize fee churn

Why 30m works:
- Captures medium-term moves (2-5 day holds)
- Less noise than 5m/15m, more trades than 4h/1d
- Regime-adaptive handles both 2021 bull and 2022 bear
- 4h/1d HTF filters prevent counter-trend disasters

Timeframe: 30m (REQUIRED for this experiment)
HTF: 4h + 1d via mtf_data helper (call ONCE before loop)
Position sizing: 0.30 discrete (0.15 in transition)
Stoploss: 2.5 * ATR(14) trailing
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_30m_regime_adaptive_4h_hma_1d_adx_kama_rsi_bb_atr_v1"
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

def calculate_kama(close, period=21, fast=2, slow=30):
    """Calculate Kaufman Adaptive Moving Average."""
    close_s = pd.Series(close)
    n = len(close)
    kama = np.full(n, np.nan)
    
    # Efficiency Ratio
    for i in range(period, n):
        change = np.abs(close[i] - close[i-period])
        volatility = np.sum(np.abs(np.diff(close[i-period:i+1])))
        if volatility > 0:
            er = change / volatility
        else:
            er = 0
        
        fast_sc = (2 / (fast + 1)) ** 2
        slow_sc = (2 / (slow + 1)) ** 2
        sc = (er * (fast_sc - slow_sc) + slow_sc) ** 2
        
        if i == period:
            kama[i] = close[i]
        else:
            kama[i] = kama[i-1] + sc * (close[i] - kama[i-1])
    
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

def calculate_bollinger(close, period=20, std_dev=2.0):
    """Calculate Bollinger Bands."""
    close_s = pd.Series(close)
    sma = close_s.rolling(window=period, min_periods=period).mean()
    std = close_s.rolling(window=period, min_periods=period).std()
    upper = sma + std_dev * std
    lower = sma - std_dev * std
    return upper.values, lower.values, sma.values

def calculate_ema(close, period=8):
    """Calculate Exponential Moving Average."""
    close_s = pd.Series(close)
    return close_s.ewm(span=period, min_periods=period, adjust=False).mean().values

def calculate_volume_ratio(volume, period=20):
    """Calculate volume ratio vs moving average."""
    vol_s = pd.Series(volume)
    vol_ma = vol_s.rolling(window=period, min_periods=period).mean()
    return (vol_s / vol_ma).values

def generate_signals(prices):
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    volume = prices["volume"].values
    n = len(close)
    
    # Load HTF data ONCE before loop (Rule 1 - CRITICAL)
    df_4h = get_htf_data(prices, '4h')
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate 4h HMA for trend bias
    hma_4h = calculate_hma(df_4h['close'].values, 21)
    hma_4h_aligned = align_htf_to_ltf(prices, df_4h, hma_4h)
    
    # Calculate 1d ADX for regime detection
    adx_1d_high = df_1d['high'].values
    adx_1d_low = df_1d['low'].values
    adx_1d_close = df_1d['close'].values
    adx_1d = calculate_adx(adx_1d_high, adx_1d_low, adx_1d_close, 14)
    adx_1d_aligned = align_htf_to_ltf(prices, df_1d, adx_1d)
    
    # Calculate 30m indicators
    atr = calculate_atr(high, low, close, 14)
    kama = calculate_kama(close, 21)
    ema_8 = calculate_ema(close, 8)
    rsi = calculate_rsi(close, 14)
    bb_upper, bb_lower, bb_mid = calculate_bollinger(close, 20, 2.0)
    vol_ratio = calculate_volume_ratio(volume, 20)
    
    signals = np.zeros(n)
    
    # Position sizing - discrete levels (Rule 4)
    SIZE_TREND = 0.30
    SIZE_RANGE = 0.30
    SIZE_TRANSITION = 0.15
    
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
        
        if np.isnan(hma_4h_aligned[i]):
            signals[i] = 0.0
            continue
        
        if np.isnan(adx_1d_aligned[i]):
            signals[i] = 0.0
            continue
        
        if np.isnan(kama[i]) or np.isnan(rsi[i]):
            signals[i] = 0.0
            continue
        
        if np.isnan(bb_upper[i]) or np.isnan(bb_lower[i]):
            signals[i] = 0.0
            continue
        
        # === 1D ADX REGIME DETECTION ===
        adx_val = adx_1d_aligned[i]
        trending_regime = adx_val > 25
        ranging_regime = adx_val < 20
        transition_regime = 20 <= adx_val <= 25
        
        # Determine position size based on regime
        if transition_regime:
            current_size = SIZE_TRANSITION
        else:
            current_size = SIZE_TREND if trending_regime else SIZE_RANGE
        
        # === 4H HMA TREND BIAS ===
        bull_trend_4h = close[i] > hma_4h_aligned[i]
        bear_trend_4h = close[i] < hma_4h_aligned[i]
        
        # === VOLUME CONFIRMATION ===
        high_volume = vol_ratio[i] > 1.5
        
        # === GENERATE SIGNAL BASED ON REGIME ===
        new_signal = 0.0
        
        if trending_regime:
            # TREND FOLLOWING: KAMA crossover + pullback to EMA8
            kama_bull = close[i] > kama[i] and kama[i] > kama[i-1] if not np.isnan(kama[i-1]) else False
            kama_bear = close[i] < kama[i] and kama[i] < kama[i-1] if not np.isnan(kama[i-1]) else False
            
            # Pullback entry (price near EMA8 in trend direction)
            pullback_long = bull_trend_4h and close[i] <= ema_8[i] * 1.005 and close[i] > kama[i]
            pullback_short = bear_trend_4h and close[i] >= ema_8[i] * 0.995 and close[i] < kama[i]
            
            if kama_bull and bull_trend_4h and (pullback_long or high_volume):
                new_signal = current_size
            elif kama_bear and bear_trend_4h and (pullback_short or high_volume):
                new_signal = -current_size
        
        elif ranging_regime:
            # MEAN REVERSION: RSI extremes + Bollinger Band touch
            rsi_long = rsi[i] < 35 and close[i] <= bb_lower[i] * 1.002
            rsi_short = rsi[i] > 65 and close[i] >= bb_upper[i] * 0.998
            
            # Must align with 4h trend bias for safety
            if rsi_long and bull_trend_4h:
                new_signal = current_size
            elif rsi_short and bear_trend_4h:
                new_signal = -current_size
        
        else:
            # TRANSITION: Reduced size, wait for clearer signals
            # Only enter on strong RSI extremes
            rsi_extreme_long = rsi[i] < 30 and bull_trend_4h
            rsi_extreme_short = rsi[i] > 70 and bear_trend_4h
            
            if rsi_extreme_long:
                new_signal = SIZE_TRANSITION
            elif rsi_extreme_short:
                new_signal = -SIZE_TRANSITION
        
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