#!/usr/bin/env python3
"""
Experiment #095: 12h KAMA Adaptive Trend + 1d HMA Filter + Volume Confirmation
Hypothesis: KAMA (Kaufman Adaptive Moving Average) adapts to market volatility better than EMA/HMA.
In trending markets, KAMA follows price closely. In choppy markets, KAMA flattens (reduces whipsaws).
This is CRITICAL for 12h timeframe where false breakouts are common.

Why this might work (learning from #089 failure Sharpe=-0.155):
- #089 used Donchian breakout + ADX>25 (too restrictive = few trades)
- #083 (12h Supertrend + 1d HMA + RSI) Sharpe=0.085 - Supertrend works on 12h!
- Key insight: KAMA adapts to volatility = fewer false signals than fixed EMA/HMA
- Volume confirmation filters out low-liquidity breakouts (common on 12h)
- Lower ADX threshold (>20 not >25) ensures enough trades on all symbols
- Asymmetric sizing: 0.30 strong signals, 0.20 weak signals (reduces fee churn)

Timeframe: 12h (REQUIRED), HTF: 1d via mtf_data helper (call ONCE before loop).
Position sizing: 0.20-0.30 discrete levels. Stoploss at 3.0*ATR (wider for 12h).
Target: Beat Sharpe=0.223 (current best #088), ensure trades on ALL symbols.
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_12h_kama_1d_hma_volume_adx_v1"
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

def calculate_kama(close, period=10, fast_period=2, slow_period=30):
    """
    Calculate Kaufman Adaptive Moving Average (KAMA).
    KAMA adapts to market volatility - follows price in trends, flattens in chop.
    ER (Efficiency Ratio) = |net change| / sum of absolute changes
    SC (Smoothing Constant) = [ER * (fast_sc - slow_sc) + slow_sc]^2
    """
    n = len(close)
    kama = np.zeros(n)
    kama[:] = np.nan
    
    # Calculate Efficiency Ratio (ER)
    er = np.zeros(n)
    for i in range(period, n):
        net_change = np.abs(close[i] - close[i - period])
        sum_changes = np.sum(np.abs(np.diff(close[i - period:i + 1])))
        if sum_changes > 0:
            er[i] = net_change / sum_changes
    
    # Smoothing constants
    fast_sc = 2.0 / (fast_period + 1)
    slow_sc = 2.0 / (slow_period + 1)
    
    # Calculate KAMA
    kama[period] = close[period]  # Initialize
    for i in range(period + 1, n):
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

def calculate_adx(high, low, close, period=14):
    """Calculate ADX (Average Directional Index) for trend strength."""
    n = len(close)
    adx = np.zeros(n)
    adx[:] = np.nan
    
    # Calculate +DM and -DM
    plus_dm = np.zeros(n)
    minus_dm = np.zeros(n)
    
    for i in range(1, n):
        plus_move = high[i] - high[i-1]
        minus_move = low[i-1] - low[i]
        
        if plus_move > minus_move and plus_move > 0:
            plus_dm[i] = plus_move
        if minus_move > plus_move and minus_move > 0:
            minus_dm[i] = minus_move
    
    # Calculate TR
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]
    
    # Smooth using Wilder's method (EMA with span=period)
    atr = pd.Series(tr).ewm(span=period, min_periods=period, adjust=False).mean().values
    plus_di = 100 * pd.Series(plus_dm).ewm(span=period, min_periods=period, adjust=False).mean().values / atr
    minus_di = 100 * pd.Series(minus_dm).ewm(span=period, min_periods=period, adjust=False).mean().values / atr
    
    # Calculate DX and ADX
    dx = np.zeros(n)
    mask = (plus_di + minus_di) > 0
    dx[mask] = 100 * np.abs(plus_di[mask] - minus_di[mask]) / (plus_di[mask] + minus_di[mask])
    
    adx = pd.Series(dx).ewm(span=period, min_periods=period, adjust=False).mean().values
    
    return adx

def calculate_ema(close, period):
    """Calculate EMA."""
    return pd.Series(close).ewm(span=period, min_periods=period, adjust=False).mean().values

def calculate_volume_sma(volume, period=20):
    """Calculate SMA of volume for volume confirmation."""
    return pd.Series(volume).rolling(window=period, min_periods=period).mean().values

def generate_signals(prices):
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    volume = prices["volume"].values
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
    kama_10 = calculate_kama(close, period=10, fast_period=2, slow_period=30)
    kama_30 = calculate_kama(close, period=30, fast_period=2, slow_period=30)
    ema_21 = calculate_ema(close, 21)
    ema_50 = calculate_ema(close, 50)
    volume_sma = calculate_volume_sma(volume, 20)
    
    signals = np.zeros(n)
    
    # Position sizing - discrete levels (Rule 4)
    SIZE_BASE = 0.20
    SIZE_STRONG = 0.30
    
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
        
        if np.isnan(adx[i]) or np.isnan(kama_10[i]) or np.isnan(kama_30[i]):
            signals[i] = 0.0
            continue
        
        # === MULTI-TIMEFRAME TREND BIAS ===
        # 1d HMA = higher timeframe trend bias (stable, slow)
        bull_trend_1d = close[i] > hma_1d_aligned[i]
        bear_trend_1d = close[i] < hma_1d_aligned[i]
        
        # === KAMA ADAPTIVE TREND ===
        # KAMA(10) > KAMA(30) = short-term trend bullish
        # KAMA adapts to volatility = fewer false signals than EMA
        kama_bullish = kama_10[i] > kama_30[i]
        kama_bearish = kama_10[i] < kama_30[i]
        
        # === EMA ALIGNMENT (secondary confirmation) ===
        ema_bullish = ema_21[i] > ema_50[i]
        ema_bearish = ema_21[i] < ema_50[i]
        
        # === ADX REGIME FILTER (lower threshold for 12h = more trades) ===
        # ADX > 20 = trending market (good for trend following)
        # ADX > 30 = strong trending market (strong signals)
        trending_market = adx[i] > 20
        strong_trend = adx[i] > 30
        
        # === VOLUME CONFIRMATION ===
        # Volume > 1.2 * volume_sma = strong volume on move
        volume_confirmed = volume[i] > 1.2 * volume_sma[i] if not np.isnan(volume_sma[i]) else False
        
        # === PRICE MOMENTUM ===
        # Price above KAMA(10) = bullish momentum
        # Price below KAMA(10) = bearish momentum
        price_above_kama = close[i] > kama_10[i]
        price_below_kama = close[i] < kama_10[i]
        
        new_signal = 0.0
        
        # === LONG ENTRY CONDITIONS (multiple paths to ensure trades) ===
        # Path 1: Strong signal - KAMA bullish + 1d bullish + strong trend + volume
        if kama_bullish and bull_trend_1d and strong_trend and volume_confirmed:
            new_signal = SIZE_STRONG
        
        # Path 2: Medium signal - KAMA bullish + 1d bullish + trending (no volume req)
        if new_signal == 0.0 and kama_bullish and bull_trend_1d and trending_market:
            new_signal = SIZE_BASE
        
        # Path 3: Weaker signal - KAMA bullish + EMA bullish + price above KAMA (ensures trades)
        if new_signal == 0.0 and kama_bullish and ema_bullish and price_above_kama:
            if bull_trend_1d or trending_market:
                new_signal = SIZE_BASE
        
        # Path 4: Fallback - KAMA bullish + 1d bullish only (maximum trade generation)
        if new_signal == 0.0 and kama_bullish and bull_trend_1d:
            if price_above_kama or ema_bullish:
                new_signal = SIZE_BASE
        
        # === SHORT ENTRY CONDITIONS (multiple paths to ensure trades) ===
        # Path 1: Strong signal - KAMA bearish + 1d bearish + strong trend + volume
        if kama_bearish and bear_trend_1d and strong_trend and volume_confirmed:
            new_signal = -SIZE_STRONG
        
        # Path 2: Medium signal - KAMA bearish + 1d bearish + trending (no volume req)
        if new_signal == 0.0 and kama_bearish and bear_trend_1d and trending_market:
            new_signal = -SIZE_BASE
        
        # Path 3: Weaker signal - KAMA bearish + EMA bearish + price below KAMA (ensures trades)
        if new_signal == 0.0 and kama_bearish and ema_bearish and price_below_kama:
            if bear_trend_1d or trending_market:
                new_signal = -SIZE_BASE
        
        # Path 4: Fallback - KAMA bearish + 1d bearish only (maximum trade generation)
        if new_signal == 0.0 and kama_bearish and bear_trend_1d:
            if price_below_kama or ema_bearish:
                new_signal = -SIZE_BASE
        
        # === STOPLOSS LOGIC (Rule 6) - 3.0 * ATR for 12h (wider stops) ===
        # Update trailing highs/lows for active positions
        if in_position and position_side > 0:
            if close[i] > highest_close:
                highest_close = close[i]
            # Trailing stop: 3.0 * ATR below highest close
            stoploss_price = highest_close - 3.0 * atr[i]
            if close[i] < stoploss_price:
                new_signal = 0.0  # Stoploss hit
        
        if in_position and position_side < 0:
            if lowest_close == 0.0 or close[i] < lowest_close:
                lowest_close = close[i]
            # Trailing stop: 3.0 * ATR above lowest close
            stoploss_price = lowest_close + 3.0 * atr[i]
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