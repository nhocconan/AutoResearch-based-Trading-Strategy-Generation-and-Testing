#!/usr/bin/env python3
"""
Experiment #127: 15m KAMA Adaptive Trend + 4h HMA Filter + Volume Confirmation + ADX + ATR Stop

Hypothesis: Adapting the BEST performing strategy (#118: mtf_4h_kama_1d_hma_adx_atr_v1, Sharpe=0.478)
for 15m timeframe with critical enhancements to avoid previous 15m failures:

Why 15m failed before (#115, #121: Sharpe=-1.9, -1.5):
- RSI pullback alone = too many false signals in chop
- Supertrend = whipsaw in 2022 crash
- No volume confirmation = fake breakouts

What this adds:
- 4h HMA(21) trend bias (proven in #118 winning strategy)
- 15m KAMA(21) adaptive trend (handles volatility changes better than EMA)
- Volume ratio > 1.5 confirms real breakouts (filters fake moves)
- ADX(14) > 20 avoids choppy whipsaw (critical for 15m noise)
- ATR(14) 2.5x trailing stop protects capital
- Discrete position sizing (0.20-0.35) limits drawdown

Why this might beat #118:
- 15m catches moves earlier than 4h entry
- Volume filter reduces false breakouts (major 15m problem)
- Same proven HTF filter (4h HMA) that worked in #118
- Faster timeframe = more trades, but volume filter keeps quality high

Timeframe: 15m (REQUIRED for this experiment)
HTF: 4h via mtf_data helper (call ONCE before loop)
Position sizing: 0.20-0.35 discrete levels
Stoploss: 2.5 * ATR(14) trailing
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_15m_kama_4h_hma_volume_adx_atr_v1"
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

def calculate_kama(close, period=21, fast=2, slow=30):
    """
    Kaufman Adaptive Moving Average (KAMA).
    Adapts smoothing based on market efficiency ratio.
    More responsive in trends, smoother in chop.
    """
    n = len(close)
    kama = np.zeros(n)
    kama[:] = np.nan
    
    if n < period + slow:
        return kama
    
    # Efficiency Ratio (ER)
    er = np.zeros(n)
    er[:] = np.nan
    
    for i in range(period, n):
        signal = np.abs(close[i] - close[i - period])
        noise = np.sum(np.abs(np.diff(close[i-period:i+1])))
        if noise > 0:
            er[i] = signal / noise
        else:
            er[i] = 0
    
    # Smoothing Constant
    sc = (er * (2/(fast+1) - 2/(slow+1)) + 2/(slow+1)) ** 2
    
    # KAMA calculation
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
    """Calculate ADX for trend strength."""
    n = len(close)
    adx = np.zeros(n)
    adx[:] = np.nan
    
    if n < period * 2:
        return adx
    
    plus_dm = np.zeros(n)
    minus_dm = np.zeros(n)
    tr = np.zeros(n)
    
    for i in range(1, n):
        plus_dm[i] = max(0, high[i] - high[i-1]) if (high[i] - high[i-1]) > (low[i-1] - low[i]) else 0
        minus_dm[i] = max(0, low[i-1] - low[i]) if (low[i-1] - low[i]) > (high[i] - high[i-1]) else 0
        tr[i] = max(high[i] - low[i], abs(high[i] - close[i-1]), abs(low[i] - close[i-1]))
    
    plus_dm_s = pd.Series(plus_dm).ewm(span=period, min_periods=period, adjust=False).mean().values
    minus_dm_s = pd.Series(minus_dm).ewm(span=period, min_periods=period, adjust=False).mean().values
    tr_s = pd.Series(tr).ewm(span=period, min_periods=period, adjust=False).mean().values
    
    plus_di = np.zeros(n)
    minus_di = np.zeros(n)
    dx = np.zeros(n)
    
    mask = tr_s > 0
    plus_di[mask] = 100 * plus_dm_s[mask] / tr_s[mask]
    minus_di[mask] = 100 * minus_dm_s[mask] / tr_s[mask]
    
    di_sum = plus_di + minus_di
    mask2 = di_sum > 0
    dx[mask2] = 100 * np.abs(plus_di[mask2] - minus_di[mask2]) / di_sum[mask2]
    
    adx_series = pd.Series(dx).ewm(span=period, min_periods=period, adjust=False).mean()
    adx = adx_series.values
    
    return adx

def calculate_volume_ratio(volume, period=20):
    """Calculate volume ratio: current volume / rolling average volume."""
    vol_s = pd.Series(volume)
    vol_avg = vol_s.rolling(window=period, min_periods=period).mean().values
    vol_ratio = volume / vol_avg
    vol_ratio[vol_avg == 0] = np.nan
    return vol_ratio

def calculate_rsi(close, period=14):
    """Calculate RSI indicator."""
    close_s = pd.Series(close)
    delta = close_s.diff()
    gain = delta.where(delta > 0, 0.0)
    loss = (-delta).where(delta < 0, 0.0)
    avg_gain = gain.ewm(span=period, min_periods=period, adjust=False).mean()
    avg_loss = loss.ewm(span=period, min_periods=period, adjust=False).mean()
    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))
    rsi[avg_loss == 0] = 100.0
    return rsi.values

def generate_signals(prices):
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    volume = prices["volume"].values
    n = len(close)
    
    # Load HTF data ONCE before loop (Rule 1 - CRITICAL)
    df_4h = get_htf_data(prices, '4h')
    
    # Calculate HTF indicators
    hma_4h = calculate_hma(df_4h['close'].values, 21)
    
    # Align HTF to LTF (Rule 2 - no manual index mapping, auto shift(1))
    hma_4h_aligned = align_htf_to_ltf(prices, df_4h, hma_4h)
    
    # Calculate 15m indicators
    atr = calculate_atr(high, low, close, 14)
    kama = calculate_kama(close, 21)
    adx = calculate_adx(high, low, close, 14)
    vol_ratio = calculate_volume_ratio(volume, 20)
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
    
    for i in range(150, n):
        # Skip if indicators not ready
        if np.isnan(atr[i]) or atr[i] == 0:
            signals[i] = 0.0
            continue
        
        if np.isnan(hma_4h_aligned[i]):
            signals[i] = 0.0
            continue
        
        if np.isnan(kama[i]) or np.isnan(adx[i]):
            signals[i] = 0.0
            continue
        
        if np.isnan(vol_ratio[i]) or np.isnan(rsi[i]):
            signals[i] = 0.0
            continue
        
        # === MULTI-TIMEFRAME TREND BIAS ===
        # 4h HMA = higher timeframe trend bias (proven in #118)
        bull_trend_4h = close[i] > hma_4h_aligned[i]
        bear_trend_4h = close[i] < hma_4h_aligned[i]
        
        # === KAMA ADAPTIVE TREND ===
        # Price above KAMA = bullish momentum
        bull_kama = close[i] > kama[i]
        bear_kama = close[i] < kama[i]
        
        # KAMA slope (momentum) - 5 bar lookback
        kama_slope = kama[i] - kama[i-5] if i >= 5 else 0
        kama_bull_slope = kama_slope > 0
        kama_bear_slope = kama_slope < 0
        
        # === ADX TREND STRENGTH ===
        adx_strong = adx[i] > 20  # Trending market (avoid chop)
        adx_weak = adx[i] <= 20   # Ranging market
        
        # === VOLUME CONFIRMATION ===
        # Volume ratio > 1.5 = strong volume (real breakout)
        vol_confirmed = vol_ratio[i] > 1.5
        vol_strong = vol_ratio[i] > 2.0
        
        # === RSI FILTER (avoid extremes for entry) ===
        # Don't enter long if RSI > 70 (overbought)
        # Don't enter short if RSI < 30 (oversold)
        rsi_ok_long = rsi[i] < 70
        rsi_ok_short = rsi[i] > 30
        
        new_signal = 0.0
        
        # === LONG ENTRY CONDITIONS ===
        # Strong: 4h bullish + KAMA bullish + KAMA slope up + ADX strong + Volume confirmed + RSI OK
        if bull_trend_4h and bull_kama and kama_bull_slope and adx_strong and vol_confirmed and rsi_ok_long:
            new_signal = SIZE_STRONG if vol_strong else SIZE_BASE
        # Moderate: 4h bullish + KAMA bullish + ADX strong + Volume confirmed
        elif bull_trend_4h and bull_kama and adx_strong and vol_confirmed:
            new_signal = SIZE_BASE
        # Weak (ensure trades): 4h bullish + KAMA bullish + Volume confirmed
        elif bull_trend_4h and bull_kama and vol_confirmed:
            new_signal = SIZE_BASE
        
        # === SHORT ENTRY CONDITIONS ===
        # Strong: 4h bearish + KAMA bearish + KAMA slope down + ADX strong + Volume confirmed + RSI OK
        if bear_trend_4h and bear_kama and kama_bear_slope and adx_strong and vol_confirmed and rsi_ok_short:
            new_signal = -SIZE_STRONG if vol_strong else -SIZE_BASE
        # Moderate: 4h bearish + KAMA bearish + ADX strong + Volume confirmed
        elif bear_trend_4h and bear_kama and adx_strong and vol_confirmed:
            new_signal = -SIZE_BASE
        # Weak (ensure trades): 4h bearish + KAMA bearish + Volume confirmed
        elif bear_trend_4h and bear_kama and vol_confirmed:
            new_signal = -SIZE_BASE
        
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