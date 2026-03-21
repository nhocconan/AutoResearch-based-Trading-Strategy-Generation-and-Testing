#!/usr/bin/env python3
"""
EXPERIMENT #023 - Dual HTF Trend Alignment + Pullback Entry (12h primary, 1d/1w HTF)
====================================================================================
Hypothesis: Most failed strategies had conflicting signals across timeframes. This strategy
requires BOTH 1d AND 1w trends to align before taking 12h pullback entries. This triple
timeframe alignment should dramatically reduce false signals and whipsaws.

Key features:
- Primary TF: 12h (required for this experiment)
- HTF filters: 1d HMA(50) + 1w HMA(21) - BOTH must agree on trend direction
- Entry trigger: Price pullback to 12h EMA(21) in trend direction
- Confirmation: RSI(14) not extreme (>30 for long, <70 for short)
- Regime filter: ADX(14) > 25 (trending market only)
- Stoploss: 2.5*ATR(14) trailing
- Position sizing: 0.30 discrete (30% of capital)
- Take profit: Reduce to half at 2R, trail stop at 1R

Why this differs from failures:
- #017 donchian_adx_htf_12h_1d_1w_v1 failed with Sharpe=-0.209, but used Donchian breakouts
- This uses pullback entries (buy dips in uptrend) which have better risk/reward
- Dual HTF alignment (1d + 1w) is stricter than single HTF filter
- ADX regime filter ensures we only trade in trending markets
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "dual_htf_trend_pullback_12h_1d_1w_v1"
timeframe = "12h"
leverage = 1.0


def calculate_hma(close, period):
    """Calculate Hull Moving Average"""
    close_s = pd.Series(close)
    wma1 = close_s.ewm(span=period // 2, adjust=False).mean()
    wma2 = close_s.ewm(span=period, adjust=False).mean()
    raw_hma = 2 * wma1 - wma2
    hma = raw_hma.ewm(span=int(np.sqrt(period)), adjust=False).mean()
    return hma.values


def calculate_ema(close, period):
    """Calculate Exponential Moving Average"""
    close_s = pd.Series(close)
    ema = close_s.ewm(span=period, adjust=False, min_periods=period).mean()
    return ema.values


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
    
    # Calculate +DM and -DM
    plus_dm = np.zeros(n)
    minus_dm = np.zeros(n)
    
    for i in range(1, n):
        plus_move = high[i] - high[i - 1]
        minus_move = low[i - 1] - low[i]
        
        if plus_move > minus_move and plus_move > 0:
            plus_dm[i] = plus_move
        if minus_move > plus_move and minus_move > 0:
            minus_dm[i] = minus_move
    
    # Calculate ATR
    atr = calculate_atr(high, low, close, period)
    
    # Calculate +DI and -DI
    plus_di = np.zeros(n)
    minus_di = np.zeros(n)
    
    plus_dm_smooth = pd.Series(plus_dm).ewm(span=period, adjust=False, min_periods=period).mean().values
    minus_dm_smooth = pd.Series(minus_dm).ewm(span=period, adjust=False, min_periods=period).mean().values
    
    for i in range(n):
        if atr[i] > 0:
            plus_di[i] = 100 * plus_dm_smooth[i] / atr[i]
            minus_di[i] = 100 * minus_dm_smooth[i] / atr[i]
    
    # Calculate DX and ADX
    dx = np.zeros(n)
    for i in range(n):
        di_sum = plus_di[i] + minus_di[i]
        if di_sum > 0:
            dx[i] = 100 * abs(plus_di[i] - minus_di[i]) / di_sum
    
    adx = pd.Series(dx).ewm(span=period, adjust=False, min_periods=period).mean().values
    
    return adx


def calculate_zscore(close, period=20):
    """Calculate Z-score for mean reversion filter"""
    close_s = pd.Series(close)
    sma = close_s.rolling(window=period, min_periods=period).mean()
    std = close_s.rolling(window=period, min_periods=period).std()
    zscore = (close_s - sma) / (std + 1e-10)
    return zscore.values


def generate_signals(prices: pd.DataFrame) -> np.ndarray:
    close = prices["close"].values.copy()
    high = prices["high"].values.copy()
    low = prices["low"].values.copy()
    n = len(close)
    
    # Load HTF data ONCE before loop (Rule 1)
    df_1d = get_htf_data(prices, '1d')
    df_1w = get_htf_data(prices, '1w')
    
    # Calculate HTF indicators
    hma_1d = calculate_hma(df_1d['close'].values, 50)
    hma_1w = calculate_hma(df_1w['close'].values, 21)
    
    # Align HTF to LTF (auto shift(1) for completed bars only)
    hma_1d_aligned = align_htf_to_ltf(prices, df_1d, hma_1d)
    hma_1w_aligned = align_htf_to_ltf(prices, df_1w, hma_1w)
    
    # Calculate 12h indicators
    ema_12h = calculate_ema(close, 21)
    atr = calculate_atr(high, low, close, 14)
    rsi = calculate_rsi(close, 14)
    adx = calculate_adx(high, low, close, 14)
    zscore = calculate_zscore(close, 20)
    
    # Generate signals
    signals = np.zeros(n)
    SIZE = 0.30  # Base position size (30% of capital)
    HALF_SIZE = SIZE / 2  # For take profit reduction
    
    # Track position state for stoploss and take profit
    position_side = 0  # 0=flat, 1=long, -1=short
    highest_since_entry = 0.0
    lowest_since_entry = float('inf')
    entry_price = 0.0
    entry_atr = 0.0
    profit_target_hit = False
    
    min_period = 150  # Wait for all indicators to stabilize
    
    for i in range(min_period, n):
        # Check for NaN in any indicator
        if (np.isnan(hma_1d_aligned[i]) or np.isnan(hma_1w_aligned[i]) or
            np.isnan(ema_12h[i]) or np.isnan(atr[i]) or np.isnan(rsi[i]) or 
            np.isnan(adx[i]) or np.isnan(zscore[i]) or atr[i] == 0):
            signals[i] = 0.0
            continue
        
        # ===== HTF TREND ALIGNMENT (BOTH 1d AND 1w MUST AGREE) =====
        daily_trend = 1 if close[i] > hma_1d_aligned[i] else -1
        weekly_trend = 1 if close[i] > hma_1w_aligned[i] else -1
        
        # Only trade when both HTF agree
        htf_aligned = (daily_trend == weekly_trend)
        major_trend = daily_trend if htf_aligned else 0
        
        # ===== REGIME FILTER (ADX > 25 = trending market) =====
        regime_valid = adx[i] > 25
        
        # ===== PULLBACK ENTRY TRIGGER =====
        # Long: price pulled back to/near EMA in uptrend
        # Short: price rallied to/near EMA in downtrend
        price_vs_ema = (close[i] - ema_12h[i]) / ema_12h[i] * 100  # % deviation from EMA
        
        # Entry conditions
        long_pullback = (major_trend == 1 and price_vs_ema <= 2.0 and price_vs_ema >= -3.0)
        short_pullback = (major_trend == -1 and price_vs_ema >= -2.0 and price_vs_ema <= 3.0)
        
        # RSI confirmation (not oversold for long, not overbought for short)
        rsi_long_ok = rsi[i] > 30 and rsi[i] < 70
        rsi_short_ok = rsi[i] > 30 and rsi[i] < 70
        
        # Z-score filter (avoid extreme extensions)
        zscore_ok = abs(zscore[i]) < 2.0
        
        # Determine target signal
        target_signal = 0.0
        
        if htf_aligned and regime_valid and rsi_long_ok and zscore_ok and long_pullback:
            target_signal = SIZE
        elif htf_aligned and regime_valid and rsi_short_ok and zscore_ok and short_pullback:
            target_signal = -SIZE
        
        # ===== STOPLOSS AND TAKE PROFIT LOGIC =====
        stoploss_triggered = False
        take_profit_triggered = False
        trend_reversal = False
        
        if position_side != 0:
            if position_side == 1:
                # Long position - update highest
                highest_since_entry = max(highest_since_entry, close[i])
                trailing_stop = highest_since_entry - 2.5 * entry_atr
                
                # Check stoploss
                if close[i] < trailing_stop:
                    stoploss_triggered = True
                
                # Check take profit (2R from entry, where R = 2.5*ATR)
                if not profit_target_hit and entry_atr > 0:
                    if close[i] >= entry_price + 5.0 * entry_atr:
                        take_profit_triggered = True
                
                # Check trend reversal (HTF no longer aligned long)
                if major_trend == -1 or (daily_trend == -1 and weekly_trend == -1):
                    trend_reversal = True
                    
            else:
                # Short position - update lowest
                lowest_since_entry = min(lowest_since_entry, close[i])
                trailing_stop = lowest_since_entry + 2.5 * entry_atr
                
                # Check stoploss
                if close[i] > trailing_stop:
                    stoploss_triggered = True
                
                # Check take profit
                if not profit_target_hit and entry_atr > 0:
                    if close[i] <= entry_price - 5.0 * entry_atr:
                        take_profit_triggered = True
                
                # Check trend reversal
                if major_trend == 1 or (daily_trend == 1 and weekly_trend == 1):
                    trend_reversal = True
        
        # Apply signals based on conditions
        if stoploss_triggered:
            signals[i] = 0.0
            position_side = 0
            highest_since_entry = 0.0
            lowest_since_entry = float('inf')
            entry_price = 0.0
            entry_atr = 0.0
            profit_target_hit = False
            
        elif take_profit_triggered:
            # Reduce position to half at 2R profit
            signals[i] = HALF_SIZE * position_side
            profit_target_hit = True
            
        elif trend_reversal:
            # Exit on major trend reversal
            signals[i] = 0.0
            position_side = 0
            highest_since_entry = 0.0
            lowest_since_entry = float('inf')
            entry_price = 0.0
            entry_atr = 0.0
            profit_target_hit = False
            
        else:
            # Check for new entry or maintain position
            if target_signal != 0.0 and position_side == 0:
                # New entry
                signals[i] = target_signal
                position_side = 1 if target_signal > 0 else -1
                highest_since_entry = close[i]
                lowest_since_entry = close[i]
                entry_price = close[i]
                entry_atr = atr[i]
                profit_target_hit = False
            elif position_side != 0:
                # Maintain existing position
                if profit_target_hit:
                    signals[i] = HALF_SIZE * position_side
                else:
                    signals[i] = SIZE * position_side
            else:
                signals[i] = 0.0
    
    return signals