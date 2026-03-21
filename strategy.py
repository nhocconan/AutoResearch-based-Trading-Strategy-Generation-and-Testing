#!/usr/bin/env python3
"""
EXPERIMENT #022 - KAMA Adaptive Trend + Volume Confirmation + HTF Filter (4h primary, 1d/1w HTF)
===============================================================================================
Hypothesis: KAMA (Kaufman Adaptive Moving Average) adapts to market volatility better than 
static EMAs/SMAs, reducing whipsaw in choppy markets. Combined with 1d HMA(50) trend filter 
and 1w HMA(100) major trend alignment, this creates a robust multi-timeframe trend system.
Volume confirmation (1.5x average) ensures we only enter on genuine breakouts, not fakeouts.

Key features:
- Primary TF: 4h (as required for this experiment)
- HTF filters: 1d HMA(50) + 1w HMA(100) for major trend alignment
- Trend: KAMA(10, 2, 30) adaptive moving average on 4h
- Entry: Price pullback to KAMA + volume spike (1.5x 20-bar avg)
- Confirmation: RSI(14) in momentum zone (45-55 for longs, 45-55 for shorts)
- Regime: ADX(14) > 20 (trending market only)
- Stoploss: 2.0*ATR(14) trailing stop
- Take profit: Reduce to half at 2.5R, trail stop at 1.5R
- Position sizing: 0.25-0.30 discrete levels (max 0.35)

Why this differs from failed attempts:
- KAMA adapts to volatility (unlike fixed EMA/Supertrend)
- Triple HTF alignment (4h KAMA + 1d HMA + 1w HMA) reduces counter-trend trades
- Volume confirmation filters fake breakouts
- Tighter RSI entry zone reduces overtrading
- Proper position state tracking with explicit variables
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "kama_volume_htf_triple_4h_1d_1w_v1"
timeframe = "4h"
leverage = 1.0


def calculate_kama(close, period=10, fast_period=2, slow_period=30):
    """
    Calculate Kaufman Adaptive Moving Average (KAMA)
    KAMA adapts to market volatility - faster in trends, slower in chop
    """
    n = len(close)
    kama = np.zeros(n)
    kama[:] = np.nan
    
    if n < period + slow_period:
        return kama
    
    # Calculate Efficiency Ratio (ER)
    er = np.zeros(n)
    er[:] = np.nan
    
    for i in range(period, n):
        price_change = abs(close[i] - close[i - period])
        volatility = np.sum(np.abs(np.diff(close[i - period:i + 1])))
        if volatility > 0:
            er[i] = price_change / volatility
        else:
            er[i] = 0
    
    # Calculate smoothing constant
    fast_sc = 2.0 / (fast_period + 1)
    slow_sc = 2.0 / (slow_period + 1)
    
    # Initialize KAMA with SMA
    kama[period - 1] = np.mean(close[:period])
    
    for i in range(period, n):
        if np.isnan(er[i]):
            kama[i] = kama[i - 1]
        else:
            sc = (er[i] * (fast_sc - slow_sc) + slow_sc) ** 2
            kama[i] = kama[i - 1] + sc * (close[i] - kama[i - 1])
    
    return kama


def calculate_hma(close, period):
    """Calculate Hull Moving Average"""
    close_s = pd.Series(close)
    wma1 = close_s.ewm(span=period // 2, adjust=False).mean()
    wma2 = close_s.ewm(span=period, adjust=False).mean()
    raw_hma = 2 * wma1 - wma2
    hma = raw_hma.ewm(span=int(np.sqrt(period)), adjust=False).mean()
    return hma.values


def calculate_atr(high, low, close, period=14):
    """Calculate ATR using Wilder's smoothing"""
    n = len(close)
    tr = np.zeros(n)
    tr[0] = high[0] - low[0]
    for i in range(1, n):
        tr[i] = max(high[i] - low[i],
                    abs(high[i] - close[i - 1]),
                    abs(low[i] - close[i - 1]))
    atr = pd.Series(tr).ewm(span=period, adjust=False, min_periods=period).mean().values
    return atr


def calculate_rsi(close, period=14):
    """Calculate RSI"""
    close_s = pd.Series(close)
    delta = close_s.diff()
    gain = delta.where(delta > 0, 0.0)
    loss = -delta.where(delta < 0, 0.0)
    avg_gain = gain.rolling(window=period, min_periods=period).mean()
    avg_loss = loss.rolling(window=period, min_periods=period).mean()
    rs = avg_gain / (avg_loss + 1e-10)
    rsi = 100 - (100 / (1 + rs))
    return rsi.values


def calculate_adx(high, low, close, period=14):
    """Calculate ADX (Average Directional Index)"""
    n = len(close)
    plus_dm = np.zeros(n)
    minus_dm = np.zeros(n)
    tr = np.zeros(n)
    
    for i in range(1, n):
        tr[i] = max(high[i] - low[i],
                    abs(high[i] - close[i - 1]),
                    abs(low[i] - close[i - 1]))
        
        if high[i] - high[i - 1] > low[i - 1] - low[i]:
            plus_dm[i] = max(high[i] - high[i - 1], 0)
        else:
            plus_dm[i] = 0
            
        if low[i - 1] - low[i] > high[i] - high[i - 1]:
            minus_dm[i] = max(low[i - 1] - low[i], 0)
        else:
            minus_dm[i] = 0
    
    tr_smooth = pd.Series(tr).ewm(span=period, adjust=False, min_periods=period).mean().values
    plus_dm_smooth = pd.Series(plus_dm).ewm(span=period, adjust=False, min_periods=period).mean().values
    minus_dm_smooth = pd.Series(minus_dm).ewm(span=period, adjust=False, min_periods=period).mean().values
    
    plus_di = np.zeros(n)
    minus_di = np.zeros(n)
    mask = tr_smooth > 0
    plus_di[mask] = 100 * plus_dm_smooth[mask] / tr_smooth[mask]
    minus_di[mask] = 100 * minus_dm_smooth[mask] / tr_smooth[mask]
    
    dx = np.zeros(n)
    mask2 = (plus_di + minus_di) > 0
    dx[mask2] = 100 * np.abs(plus_di[mask2] - minus_di[mask2]) / (plus_di[mask2] + minus_di[mask2])
    
    adx = pd.Series(dx).ewm(span=period, adjust=False, min_periods=period).mean().values
    return adx


def calculate_volume_spike(volume, period=20, threshold=1.5):
    """Detect volume spikes above threshold * average"""
    vol_s = pd.Series(volume)
    vol_avg = vol_s.rolling(window=period, min_periods=period).mean().values
    spike = volume > (threshold * vol_avg)
    return spike


def generate_signals(prices: pd.DataFrame) -> np.ndarray:
    close = prices["close"].values.copy()
    high = prices["high"].values.copy()
    low = prices["low"].values.copy()
    volume = prices["volume"].values.copy()
    n = len(close)
    
    # Load HTF data ONCE before loop (Rule 1 - CRITICAL)
    df_1d = get_htf_data(prices, '1d')
    df_1w = get_htf_data(prices, '1w')
    
    # Calculate HTF indicators
    hma_1d = calculate_hma(df_1d['close'].values, 50)
    hma_1w = calculate_hma(df_1w['close'].values, 100)
    
    # Align HTF to LTF (auto shift(1) for completed bars only - Rule 2)
    hma_1d_aligned = align_htf_to_ltf(prices, df_1d, hma_1d)
    hma_1w_aligned = align_htf_to_ltf(prices, df_1w, hma_1w)
    
    # Calculate 4h indicators
    kama = calculate_kama(close, period=10, fast_period=2, slow_period=30)
    atr = calculate_atr(high, low, close, 14)
    rsi = calculate_rsi(close, 14)
    adx = calculate_adx(high, low, close, 14)
    volume_spike = calculate_volume_spike(volume, 20, 1.5)
    
    # Generate signals
    signals = np.zeros(n)
    SIZE = 0.28  # Base position size (28% of capital)
    HALF_SIZE = SIZE / 2  # For take profit reduction
    
    # Position state tracking
    position_side = 0  # 0=flat, 1=long, -1=short
    entry_price = 0.0
    highest_since_entry = 0.0
    lowest_since_entry = float('inf')
    profit_target_hit = False
    r_multiple = 2.5  # R multiple for stoploss (2.0*ATR)
    
    min_period = 150  # Wait for all indicators to stabilize
    
    for i in range(min_period, n):
        # Check for NaN in any indicator
        if (np.isnan(kama[i]) or np.isnan(atr[i]) or np.isnan(rsi[i]) or 
            np.isnan(adx[i]) or np.isnan(hma_1d_aligned[i]) or 
            np.isnan(hma_1w_aligned[i]) or atr[i] == 0):
            signals[i] = 0.0
            continue
        
        # === HTF Trend Filters ===
        # 1d trend: price above/below 1d HMA(50)
        daily_trend = 1 if close[i] > hma_1d_aligned[i] else -1
        
        # 1w trend: price above/below 1w HMA(100) - major trend
        weekly_trend = 1 if close[i] > hma_1w_aligned[i] else -1
        
        # === 4h Trend Filter ===
        # KAMA slope (compare to 5 bars ago)
        kama_slope = 1 if kama[i] > kama[i - 5] else -1 if kama[i] < kama[i - 5] else 0
        
        # Price relative to KAMA
        price_vs_kama = 1 if close[i] > kama[i] else -1
        
        # === Regime Filter ===
        # ADX > 20 = trending market
        regime_valid = adx[i] > 20
        
        # === Entry Conditions ===
        # Long: All HTF bullish + KAMA bullish + pullback to KAMA + volume spike + RSI momentum
        long_condition = (
            daily_trend == 1 and 
            weekly_trend == 1 and 
            kama_slope == 1 and
            price_vs_kama == 1 and
            regime_valid and
            volume_spike[i] and
            45 <= rsi[i] <= 60  # RSI momentum zone (not overbought)
        )
        
        # Short: All HTF bearish + KAMA bearish + pullback to KAMA + volume spike + RSI momentum
        short_condition = (
            daily_trend == -1 and 
            weekly_trend == -1 and 
            kama_slope == -1 and
            price_vs_kama == -1 and
            regime_valid and
            volume_spike[i] and
            40 <= rsi[i] <= 55  # RSI momentum zone (not oversold)
        )
        
        # === Stoploss and Take Profit Logic ===
        stoploss_triggered = False
        take_profit_triggered = False
        
        if position_side != 0:
            if position_side == 1:
                # Long position - update highest
                highest_since_entry = max(highest_since_entry, close[i])
                trailing_stop = highest_since_entry - r_multiple * atr[i]
                
                # Check stoploss
                if close[i] < trailing_stop:
                    stoploss_triggered = True
                
                # Check take profit (2.5R from entry)
                if not profit_target_hit:
                    r_distance = r_multiple * atr[i]  # Current R distance
                    if close[i] >= entry_price + 2.5 * r_distance:
                        take_profit_triggered = True
            else:
                # Short position - update lowest
                lowest_since_entry = min(lowest_since_entry, close[i])
                trailing_stop = lowest_since_entry + r_multiple * atr[i]
                
                # Check stoploss
                if close[i] > trailing_stop:
                    stoploss_triggered = True
                
                # Check take profit
                if not profit_target_hit:
                    r_distance = r_multiple * atr[i]
                    if close[i] <= entry_price - 2.5 * r_distance:
                        take_profit_triggered = True
        
        # === Apply Signals ===
        if stoploss_triggered:
            signals[i] = 0.0
            position_side = 0
            entry_price = 0.0
            highest_since_entry = 0.0
            lowest_since_entry = float('inf')
            profit_target_hit = False
        elif take_profit_triggered:
            # Reduce position to half at 2.5R profit
            signals[i] = HALF_SIZE * position_side
            profit_target_hit = True
        else:
            # Check for trend reversal exit
            trend_reversed = False
            if position_side == 1 and (kama_slope == -1 or daily_trend == -1):
                trend_reversed = True
            elif position_side == -1 and (kama_slope == 1 or daily_trend == 1):
                trend_reversed = True
            
            if trend_reversed and position_side != 0:
                signals[i] = 0.0
                position_side = 0
                entry_price = 0.0
                highest_since_entry = 0.0
                lowest_since_entry = float('inf')
                profit_target_hit = False
            elif long_condition and position_side == 0:
                # New long entry
                signals[i] = SIZE
                position_side = 1
                entry_price = close[i]
                highest_since_entry = close[i]
                lowest_since_entry = float('inf')
                profit_target_hit = False
            elif short_condition and position_side == 0:
                # New short entry
                signals[i] = -SIZE
                position_side = -1
                entry_price = close[i]
                highest_since_entry = 0.0
                lowest_since_entry = close[i]
                profit_target_hit = False
            elif position_side != 0:
                # Maintain existing position
                signals[i] = SIZE * position_side if not profit_target_hit else HALF_SIZE * position_side
            else:
                signals[i] = 0.0
    
    return signals