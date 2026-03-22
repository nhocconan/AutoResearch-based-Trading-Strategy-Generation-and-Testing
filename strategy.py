#!/usr/bin/env python3
"""
Experiment #439: 15m KAMA Adaptive Trend with 4h HMA Bias + 1h ADX Regime

Hypothesis: After 438 experiments, 15m strategies fail due to whipsaws in choppy 
markets. KAMA (Kaufman Adaptive Moving Average) adapts to volatility - fast in 
trends, slow in ranges. Combined with proven 4h HMA trend bias and 1h ADX regime 
filter, this should reduce false signals while maintaining trade frequency.

Key innovations:
1. KAMA(10) on 15m - adapts to market efficiency, reduces whipsaws
2. 4h HMA(21) - proven trend bias from best strategies (Sharpe=0.676 baseline)
3. 1h ADX(14) - regime filter (ADX>25=trend, ADX<20=range)
4. RSI(7) confirmation - prevents false KAMA crossovers
5. Asymmetric sizing: 0.25 long, 0.20 short (bear market bias for 2025 test)
6. Stoploss: 2.0 * ATR(14) trailing

Why 15m might work now:
- KAMA adapts better than EMA to 15m noise
- Dual HTF filter (4h trend + 1h regime) proven in #432
- Looser RSI threshold (35/65 vs 30/70) ensures trade frequency
- Conservative sizing (0.20-0.25) controls 2022-style crash risk

Timeframe: 15m (REQUIRED for this experiment)
HTF: 4h HMA trend + 1h ADX regime (both via mtf_data helper)
Position sizing: 0.25 long / 0.20 short discrete
Stoploss: 2.0 * ATR(14) trailing
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_15m_kama_adaptive_4h_hma_1h_adx_rsi_atr_v1"
timeframe = "15m"
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

def calculate_kama(close, period=10, fast_period=2, slow_period=30):
    """
    Calculate Kaufman Adaptive Moving Average (KAMA).
    KAMA adapts to market volatility - faster in trends, slower in ranges.
    ER (Efficiency Ratio) = |net change| / sum of absolute changes
    SC (Smoothing Constant) = [ER * (fast_SC - slow_SC) + slow_SC]^2
    """
    n = len(close)
    kama = np.full(n, np.nan)
    
    # Efficiency Ratio calculation
    er = np.full(n, np.nan)
    for i in range(period, n):
        net_change = np.abs(close[i] - close[i - period])
        sum_changes = np.sum(np.abs(np.diff(close[i-period:i+1])))
        if sum_changes > 1e-10:
            er[i] = net_change / sum_changes
        else:
            er[i] = 0
    
    # Smoothing constants
    fast_sc = 2.0 / (fast_period + 1)
    slow_sc = 2.0 / (slow_period + 1)
    
    # KAMA calculation
    sc = np.zeros(n)
    for i in range(period, n):
        if not np.isnan(er[i]):
            sc[i] = (er[i] * (fast_sc - slow_sc) + slow_sc) ** 2
    
    # Initialize KAMA
    kama[period] = close[period]
    
    for i in range(period + 1, n):
        if not np.isnan(sc[i]):
            kama[i] = kama[i-1] + sc[i] * (close[i] - kama[i-1])
        else:
            kama[i] = kama[i-1]
    
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

def calculate_rsi(close, period=7):
    """Calculate Relative Strength Index (shorter period for 15m)."""
    close_s = pd.Series(close)
    delta = close_s.diff()
    gain = delta.where(delta > 0, 0.0)
    loss = -delta.where(delta < 0, 0.0)
    
    avg_gain = gain.ewm(span=period, min_periods=period, adjust=False).mean()
    avg_loss = loss.ewm(span=period, min_periods=period, adjust=False).mean()
    
    rs = avg_gain / avg_loss.replace(0, np.inf)
    rsi = 100 - (100 / (1 + rs))
    return rsi.values

def calculate_sma(close, period=50):
    """Calculate Simple Moving Average."""
    close_s = pd.Series(close)
    return close_s.rolling(window=period, min_periods=period).mean().values

def generate_signals(prices):
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    n = len(close)
    
    # Load HTF data ONCE before loop (Rule 1 - CRITICAL)
    df_4h = get_htf_data(prices, '4h')
    df_1h = get_htf_data(prices, '1h')
    
    # Calculate HTF indicators
    hma_4h = calculate_hma(df_4h['close'].values, 21)
    adx_1h = calculate_adx(
        df_1h['high'].values, 
        df_1h['low'].values, 
        df_1h['close'].values, 
        14
    )
    
    # Align HTF to LTF (Rule 2 - auto shift(1) for completed bars)
    hma_4h_aligned = align_htf_to_ltf(prices, df_4h, hma_4h)
    adx_1h_aligned = align_htf_to_ltf(prices, df_1h, adx_1h)
    
    # Calculate 15m indicators
    atr = calculate_atr(high, low, close, 14)
    kama = calculate_kama(close, period=10)
    rsi = calculate_rsi(close, 7)
    sma_50 = calculate_sma(close, 50)
    
    signals = np.zeros(n)
    
    # Position sizing - discrete levels (Rule 4)
    SIZE_LONG = 0.25  # Slightly larger for longs
    SIZE_SHORT = 0.20  # Smaller for shorts (bear market bias)
    
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
        
        if np.isnan(adx_1h_aligned[i]):
            signals[i] = 0.0
            continue
        
        if np.isnan(kama[i]) or np.isnan(rsi[i]):
            signals[i] = 0.0
            continue
        
        if np.isnan(sma_50[i]):
            signals[i] = 0.0
            continue
        
        # === 4H HMA TREND BIAS (proven edge) ===
        bull_trend_4h = close[i] > hma_4h_aligned[i]
        bear_trend_4h = close[i] < hma_4h_aligned[i]
        
        # === 1H ADX REGIME FILTER ===
        adx_val = adx_1h_aligned[i]
        trending_regime = adx_val > 25
        ranging_regime = adx_val < 20
        neutral_regime = not trending_regime and not ranging_regime
        
        # === KAMA SIGNAL (adaptive trend) ===
        # KAMA crossover with price
        kama_long = close[i] > kama[i] and close[i-1] <= kama[i-1] if i > 0 else close[i] > kama[i]
        kama_short = close[i] < kama[i] and close[i-1] >= kama[i-1] if i > 0 else close[i] < kama[i]
        
        # KAMA slope
        kama_slope_up = kama[i] > kama[i-1] if i > 0 else False
        kama_slope_down = kama[i] < kama[i-1] if i > 0 else False
        
        # === RSI CONFIRMATION (looser thresholds for trade frequency) ===
        rsi_long = rsi[i] < 40  # Oversold (looser than 30)
        rsi_short = rsi[i] > 60  # Overbought (looser than 70)
        rsi_neutral = 35 <= rsi[i] <= 65
        
        # === SMA FILTER (avoid counter-trend) ===
        above_sma = close[i] > sma_50[i]
        below_sma = close[i] < sma_50[i]
        
        # === GENERATE SIGNAL ===
        new_signal = 0.0
        
        # LONG ENTRY: KAMA crossover + RSI confirmation + 4H trend + regime filter
        if bull_trend_4h and above_sma:
            # In trending regime: KAMA breakout
            if trending_regime and kama_long and kama_slope_up:
                new_signal = SIZE_LONG
            # In ranging regime: RSI mean reversion
            elif ranging_regime and rsi_long and kama_slope_up:
                new_signal = SIZE_LONG
            # Neutral regime: require both signals
            elif neutral_regime and kama_long and rsi_long:
                new_signal = SIZE_LONG
        
        # SHORT ENTRY: KAMA crossover + RSI confirmation + 4H trend + regime filter
        if bear_trend_4h and below_sma:
            # In trending regime: KAMA breakdown
            if trending_regime and kama_short and kama_slope_down:
                new_signal = -SIZE_SHORT
            # In ranging regime: RSI mean reversion
            elif ranging_regime and rsi_short and kama_slope_down:
                new_signal = -SIZE_SHORT
            # Neutral regime: require both signals
            elif neutral_regime and kama_short and rsi_short:
                new_signal = -SIZE_SHORT
        
        # === STOPLOSS LOGIC (Rule 6) - 2.0 * ATR trailing ===
        if in_position and position_side != 0:
            if position_side > 0:
                # Update highest close for long position
                if close[i] > highest_close:
                    highest_close = close[i]
                stoploss_price = highest_close - 2.0 * atr[i]
                if close[i] < stoploss_price:
                    new_signal = 0.0
            
            if position_side < 0:
                # Update lowest close for short position
                if lowest_close == 0.0 or close[i] < lowest_close:
                    lowest_close = close[i]
                stoploss_price = lowest_close + 2.0 * atr[i]
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