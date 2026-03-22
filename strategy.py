#!/usr/bin/env python3
"""
Experiment #171: 1h KAMA Adaptive Trend + 4h HMA Filter + ADX + ATR Stop

Hypothesis: KAMA (Kaufman Adaptive Moving Average) adapts to market volatility,
moving fast in trends and slow in ranges. This should reduce whipsaws compared
to fixed EMA/HMA while still capturing trend moves. Combined with 4h HMA for
higher timeframe bias and ADX for trend confirmation.

Why KAMA on 1h:
- KAMA proved effective in current best strategy (4h KAMA + 1d HMA, Sharpe=0.478)
- Adapts to crypto's changing volatility regimes automatically
- Less lag than EMA in trends, less noise than HMA in ranges
- 1h timeframe balances trade frequency with signal quality (more trades than 4h/12h)

Learning from failures:
- #163 (15m RSI): Too noisy, whipsawed in chop
- #165 (1h regime adaptive): Over-complicated with too many filters
- #166 (4h CRSI): Catastrophic failure, complex RSI variants don't work
- #170 (30m Fisher): Fisher transform too sensitive for crypto volatility
- Simple adaptive MA crossover worked on 4h, should work on 1h with HTF filter

Key improvements:
- KAMA crossover (fast vs slow) instead of fixed EMA/HMA crossover
- ADX > 15 threshold (lower than 20-25) to ensure sufficient trade count
- 4h HMA proven effective in multiple strategies
- 2.5*ATR trailing stop with RSI profit-taking exit

Timeframe: 1h (REQUIRED for experiment #171)
HTF: 4h HMA via mtf_data helper (call ONCE before loop)
Position sizing: 0.25-0.35 discrete levels
Stoploss: 2.5 * ATR(14) trailing
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_1h_kama_4h_hma_adx_atr_v1"
timeframe = "1h"
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

def calculate_hma(close, period=21):
    """Calculate Hull Moving Average for smoother trend with less lag."""
    close_s = pd.Series(close)
    half = max(1, period // 2)
    sqrt_period = max(1, int(np.sqrt(period)))
    wma1 = close_s.ewm(span=half, min_periods=half, adjust=False).mean()
    wma2 = close_s.ewm(span=period, min_periods=period, adjust=False).mean()
    wma3 = (2 * wma1 - wma2).ewm(span=sqrt_period, min_periods=sqrt_period, adjust=False).mean()
    return wma3.values

def calculate_kama(close, period=10, fast_period=2, slow_period=30):
    """
    Kaufman Adaptive Moving Average (KAMA).
    Adapts smoothing based on market efficiency ratio.
    Fast in trends, slow in ranges.
    """
    n = len(close)
    kama = np.zeros(n)
    
    # Calculate Efficiency Ratio (ER)
    change = np.abs(close - np.roll(close, period))
    change[0:period] = np.abs(close[0:period] - close[0])
    
    volatility = np.zeros(n)
    for i in range(period, n):
        vol_sum = 0.0
        for j in range(1, period + 1):
            vol_sum += np.abs(close[i-j+1] - close[i-j])
        volatility[i] = vol_sum
    
    er = np.zeros(n)
    er[period:] = change[period:] / (volatility[period:] + 1e-10)
    er[:period] = er[period]  # Fill initial values
    
    # Smoothing constant
    fast_sc = 2.0 / (fast_period + 1)
    slow_sc = 2.0 / (slow_period + 1)
    sc = (er * (fast_sc - slow_sc) + slow_sc) ** 2
    
    # KAMA calculation
    kama[0] = close[0]
    for i in range(1, n):
        kama[i] = kama[i-1] + sc[i] * (close[i] - kama[i-1])
    
    return kama

def calculate_rsi(close, period=14):
    """Calculate RSI for profit-taking exits."""
    close_s = pd.Series(close)
    delta = close_s.diff()
    gain = delta.where(delta > 0, 0.0)
    loss = -delta.where(delta < 0, 0.0)
    
    avg_gain = gain.ewm(span=period, min_periods=period, adjust=False).mean()
    avg_loss = loss.ewm(span=period, min_periods=period, adjust=False).mean()
    
    rs = avg_gain / (avg_loss + 1e-10)
    rsi = 100 - (100 / (1 + rs))
    rsi = rsi.fillna(50).values
    
    return rsi

def generate_signals(prices):
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    n = len(close)
    
    # Load HTF data ONCE before loop (CRITICAL - Rule 1)
    df_4h = get_htf_data(prices, '4h')
    
    # Calculate HTF indicators
    hma_4h = calculate_hma(df_4h['close'].values, 21)
    
    # Align HTF to LTF (CRITICAL - Rule 2, auto shift(1))
    hma_4h_aligned = align_htf_to_ltf(prices, df_4h, hma_4h)
    
    # Calculate 1h indicators
    kama_slow = calculate_kama(close, 10, 2, 30)  # Adaptive trend
    kama_fast = calculate_kama(close, 5, 2, 20)   # Faster adaptive for signals
    adx = calculate_adx(high, low, close, 14)
    atr = calculate_atr(high, low, close, 14)
    rsi = calculate_rsi(close, 14)
    
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
        
        if np.isnan(hma_4h_aligned[i]):
            signals[i] = 0.0
            continue
        
        if np.isnan(adx[i]) or np.isnan(kama_slow[i]) or np.isnan(kama_fast[i]):
            signals[i] = 0.0
            continue
        
        # === MULTI-TIMEFRAME TREND BIAS ===
        # 4h HMA = higher timeframe trend direction (proven effective)
        bull_trend_4h = close[i] > hma_4h_aligned[i]
        bear_trend_4h = close[i] < hma_4h_aligned[i]
        
        # === TREND STRENGTH FILTER ===
        # ADX > 15 = trending market (lower threshold for more trades)
        trend_strength = adx[i] > 15
        
        # === KAMA CROSSOVER SIGNAL ===
        # Fast KAMA above slow KAMA = bullish momentum
        # Fast KAMA below slow KAMA = bearish momentum
        kama_bullish = kama_fast[i] > kama_slow[i]
        kama_bearish = kama_fast[i] < kama_slow[i]
        
        # === RSI PROFIT TAKING ===
        # Exit long when RSI overbought (>70)
        # Exit short when RSI oversold (<30)
        rsi_overbought = rsi[i] > 70
        rsi_oversold = rsi[i] < 30
        
        new_signal = 0.0
        
        # === LONG ENTRY CONDITIONS ===
        # 4h bullish + ADX trending + KAMA bullish crossover
        if bull_trend_4h and trend_strength and kama_bullish:
            new_signal = SIZE_BASE
        
        # === SHORT ENTRY CONDITIONS ===
        # 4h bearish + ADX trending + KAMA bearish crossover
        if bear_trend_4h and trend_strength and kama_bearish:
            new_signal = -SIZE_BASE
        
        # === RSI PROFIT TAKING ===
        # Exit long when RSI overbought
        if in_position and position_side > 0 and rsi_overbought:
            new_signal = 0.0
        
        # Exit short when RSI oversold
        if in_position and position_side < 0 and rsi_oversold:
            new_signal = 0.0
        
        # === STOPLOSS LOGIC (Rule 6) - 2.5 * ATR trailing ===
        # Update trailing highs/lows for active positions
        if in_position and position_side > 0:
            if close[i] > highest_close:
                highest_close = close[i]
            # Trailing stop: 2.5 * ATR below highest close
            stoploss_price = highest_close - 2.5 * atr[i]
            if close[i] < stoploss_price:
                new_signal = 0.0  # Stoploss hit
        
        if in_position and position_side < 0:
            if lowest_close == 0.0 or close[i] < lowest_close:
                lowest_close = close[i]
            # Trailing stop: 2.5 * ATR above lowest close
            stoploss_price = lowest_close + 2.5 * atr[i]
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