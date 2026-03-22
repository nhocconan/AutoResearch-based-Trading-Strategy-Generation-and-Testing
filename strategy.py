#!/usr/bin/env python3
"""
Experiment #177: 1h KAMA Adaptive Trend + 12h HMA Filter + Donchian Breakout + ADX/RSI Confirmation

Hypothesis: 1h timeframe balances signal frequency and noise better than 15m/30m (too noisy) 
or 4h/12h (too few trades). KAMA adapts to volatility regimes, 12h HMA provides stable 
HTF bias, Donchian(20) breakout confirms momentum, ADX>20 filters weak trends, and RSI 
momentum (55/45 threshold) confirms direction without mean-reversion extremes.

Why 1h might work:
- More trades than 4h/12h strategies (ensures >=10 trades per symbol)
- Less noise than 15m/30m (fewer false breakouts)
- 12h HMA is stable enough to avoid whipsaws but responsive enough for crypto
- Donchian breakout captures momentum bursts common in crypto
- ADX>20 (not 25+) ensures sufficient trade count
- RSI 55/45 (not 70/30) captures momentum without waiting for extremes

Learning from failures:
- #175 (15m trend pullback): Sharpe=-4.69 - too much noise on lower TF
- #176 (30m KAMA): Sharpe=-1.54 - similar issue, 30m still too noisy
- #170 (30m Fisher): Sharpe=-1.72 - mean-reversion fails on crypto
- #171 (1h KAMA 4h HMA): Sharpe=-1.37 - needs better entry confirmation
- Mean reversion consistently fails on BTC/ETH (CRSI, RSI extremes)
- Trend following with HTF filter works best

Timeframe: 1h (REQUIRED for this experiment)
HTF: 12h via mtf_data helper (call ONCE before loop)
Position sizing: 0.25 discrete levels (conservative for drawdown control)
Stoploss: 2.5 * ATR(14) trailing
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_1h_kama_12h_hma_donchian_adx_rsi_atr_v1"
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

def calculate_kama(close, er_period=10, fast_period=2, slow_period=30):
    """Calculate Kaufman Adaptive Moving Average (KAMA)."""
    n = len(close)
    kama = np.zeros(n)
    
    net_change = np.abs(close - np.roll(close, er_period))
    net_change[:er_period] = np.abs(close[:er_period] - close[0])
    
    sum_abs_change = np.zeros(n)
    for i in range(er_period, n):
        sum_abs_change[i] = np.sum(np.abs(close[i-er_period+1:i+1] - np.roll(close[i-er_period+1:i+1], 1)))
    sum_abs_change[:er_period] = sum_abs_change[er_period]
    
    sum_abs_change = np.where(sum_abs_change == 0, 1e-10, sum_abs_change)
    er = net_change / sum_abs_change
    er = np.clip(er, 0, 1)
    
    fast_sc = 2.0 / (fast_period + 1)
    slow_sc = 2.0 / (slow_period + 1)
    
    sc = (er * (fast_sc - slow_sc) + slow_sc) ** 2
    
    kama[0] = close[0]
    for i in range(1, n):
        kama[i] = kama[i-1] + sc[i] * (close[i] - kama[i-1])
    
    return kama

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

def calculate_donchian(high, low, period=20):
    """Calculate Donchian Channel (upper/lower bounds)."""
    n = len(high)
    upper = np.zeros(n)
    lower = np.zeros(n)
    
    for i in range(period-1, n):
        upper[i] = np.max(high[i-period+1:i+1])
        lower[i] = np.min(low[i-period+1:i+1])
    
    upper[:period-1] = np.nan
    lower[:period-1] = np.nan
    
    return upper, lower

def calculate_ema(close, period=21):
    """Calculate Exponential Moving Average."""
    close_s = pd.Series(close)
    ema = close_s.ewm(span=period, min_periods=period, adjust=False).mean()
    return ema.values

def generate_signals(prices):
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    n = len(close)
    
    # Load HTF data ONCE before loop (Rule 1 - CRITICAL)
    df_12h = get_htf_data(prices, '12h')
    
    # Calculate HTF indicators
    hma_12h = calculate_hma(df_12h['close'].values, 21)
    
    # Align HTF to LTF (Rule 2 - no manual index mapping, auto shift(1))
    hma_12h_aligned = align_htf_to_ltf(prices, df_12h, hma_12h)
    
    # Calculate 1h indicators
    atr = calculate_atr(high, low, close, 14)
    adx = calculate_adx(high, low, close, 14)
    kama = calculate_kama(close, er_period=10, fast_period=2, slow_period=30)
    rsi = calculate_rsi(close, 14)
    ema_21 = calculate_ema(close, 21)
    ema_50 = calculate_ema(close, 50)
    donchian_upper, donchian_lower = calculate_donchian(high, low, 20)
    
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
        
        if np.isnan(hma_12h_aligned[i]):
            signals[i] = 0.0
            continue
        
        if np.isnan(adx[i]) or np.isnan(kama[i]) or np.isnan(rsi[i]):
            signals[i] = 0.0
            continue
        
        if np.isnan(donchian_upper[i]) or np.isnan(donchian_lower[i]):
            signals[i] = 0.0
            continue
        
        # === MULTI-TIMEFRAME TREND BIAS ===
        # 12h HMA = higher timeframe trend bias
        bull_trend_12h = close[i] > hma_12h_aligned[i]
        bear_trend_12h = close[i] < hma_12h_aligned[i]
        
        # === TREND STRENGTH FILTER ===
        # ADX > 20 = trending market (lower than 25 to ensure trades)
        trend_strength = adx[i] > 20
        
        # === KAMA ADAPTIVE TREND ===
        price_above_kama = close[i] > kama[i]
        price_below_kama = close[i] < kama[i]
        
        # === RSI MOMENTUM ===
        # RSI > 55 = bullish momentum (not extreme 70)
        # RSI < 45 = bearish momentum (not extreme 30)
        rsi_bullish = rsi[i] > 55
        rsi_bearish = rsi[i] < 45
        
        # === EMA CONFIRMATION ===
        ema_bullish = ema_21[i] > ema_50[i]
        ema_bearish = ema_21[i] < ema_50[i]
        
        # === DONCHIAN BREAKOUT ===
        # Price breaking above Donchian upper = bullish breakout
        # Price breaking below Donchian lower = bearish breakout
        donchian_bullish = close[i] > donchian_upper[i]
        donchian_bearish = close[i] < donchian_lower[i]
        
        new_signal = 0.0
        
        # === ENTRY CONDITIONS ===
        # Long: 12h bullish + ADX trending + price>KAMA + RSI>55 + (EMA or Donchian)
        # Using flexible scoring to ensure enough trades
        long_score = 0
        if bull_trend_12h:
            long_score += 2  # HTF bias is most important
        if trend_strength:
            long_score += 1
        if price_above_kama:
            long_score += 1
        if rsi_bullish:
            long_score += 1
        if ema_bullish or donchian_bullish:
            long_score += 1
        
        if long_score >= 4:
            new_signal = SIZE_BASE
        
        # Short: 12h bearish + ADX trending + price<KAMA + RSI<45 + (EMA or Donchian)
        short_score = 0
        if bear_trend_12h:
            short_score += 2
        if trend_strength:
            short_score += 1
        if price_below_kama:
            short_score += 1
        if rsi_bearish:
            short_score += 1
        if ema_bearish or donchian_bearish:
            short_score += 1
        
        if short_score >= 4:
            new_signal = -SIZE_BASE
        
        # === STOPLOSS LOGIC (Rule 6) - 2.5 * ATR trailing ===
        if in_position:
            if position_side > 0:
                if close[i] > highest_close:
                    highest_close = close[i]
                stoploss_price = highest_close - 2.5 * atr[i]
                if close[i] < stoploss_price:
                    new_signal = 0.0
            
            if position_side < 0:
                if lowest_close == 0.0 or close[i] < lowest_close:
                    lowest_close = close[i]
                stoploss_price = lowest_close + 2.5 * atr[i]
                if close[i] > stoploss_price:
                    new_signal = 0.0
        
        # === UPDATE POSITION TRACKING ===
        if new_signal != 0.0:
            if not in_position:
                in_position = True
                position_side = np.sign(new_signal)
                highest_close = close[i] if position_side > 0 else 0.0
                lowest_close = close[i] if position_side < 0 else 0.0
            elif np.sign(new_signal) != position_side:
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