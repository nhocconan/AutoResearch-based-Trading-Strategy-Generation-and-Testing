#!/usr/bin/env python3
"""
Experiment #526: 4h Fisher Transform with Dual HTF HMA + ADX Regime

Hypothesis: The Ehlers Fisher Transform excels at catching reversals in bear/range markets (2025+). 
Combined with dual HTF confirmation (1d + 1w HMA) and ADX regime filter, this should:
1. Catch bear market rallies (Fisher crosses up from extreme lows)
2. Avoid trend whipsaw (ADX hysteresis + dual HMA confirmation)
3. Generate enough trades (loose Fisher thresholds: -1.0/+1.0)
4. Control drawdown (2.0*ATR stoploss, discrete sizing 0.25)

Key innovations:
1. DUAL HTF FILTER: At least ONE of 1d/1w HMA must agree for directional bias
2. FISHER TRANSFORM: period=9, catches reversals better than RSI in bear markets
3. ADX HYSTERESIS: Enter >18, exit <14 (looser for more trades)
4. VOLATILITY REGIME: ATR(7)/ATR(21) adjusts position sizing
5. LOOSE ENTRY: Fisher < -1.0 long, > +1.0 short (ensures >=10 trades/year)

Timeframe: 4h (REQUIRED for this experiment)
HTF: 1d and 1w via mtf_data helper (call ONCE before loop)
Position sizing: 0.25 base, reduced to 0.15 in high vol
Stoploss: 2.0 * ATR(14) trailing
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_4h_fisher_dual_htf_hma_adx_regime_asymmetric_atr_v1"
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
    """Calculate Hull Moving Average."""
    close_s = pd.Series(close)
    half = max(1, period // 2)
    sqrt_period = max(1, int(np.sqrt(period)))
    wma1 = close_s.ewm(span=half, min_periods=half, adjust=False).mean()
    wma2 = close_s.ewm(span=period, min_periods=period, adjust=False).mean()
    wma3 = (2 * wma1 - wma2).ewm(span=sqrt_period, min_periods=sqrt_period, adjust=False).mean()
    return wma3.values

def calculate_fisher_transform(high, low, close, period=9):
    """
    Calculate Ehlers Fisher Transform.
    Normalizes price to -1 to +1, then applies Fisher transform to make extremes more visible.
    """
    n = len(high)
    fisher = np.full(n, np.nan)
    
    for i in range(period, n):
        highest = np.max(high[i-period+1:i+1])
        lowest = np.min(low[i-period+1:i+1])
        
        if highest == lowest:
            fisher[i] = fisher[i-1] if i > 0 and not np.isnan(fisher[i-1]) else 0
            continue
        
        price = (close[i] - lowest) / (highest - lowest)
        price = np.clip(price, 0.001, 0.999)
        fisher_val = 0.5 * np.log((1 + price) / (1 - price))
        
        if i > period and not np.isnan(fisher[i-1]):
            fisher[i] = 0.67 * fisher_val + 0.33 * fisher[i-1]
        else:
            fisher[i] = fisher_val
    
    return fisher

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

def calculate_adx_hysteresis(adx, enter_thresh=18, exit_thresh=14):
    """Calculate ADX with hysteresis to prevent whipsaw."""
    n = len(adx)
    trending = np.zeros(n)
    state = 0
    
    for i in range(len(adx)):
        if np.isnan(adx[i]):
            continue
        if state == 0 and adx[i] > enter_thresh:
            state = 1
        elif state == 1 and adx[i] < exit_thresh:
            state = 0
        trending[i] = state
    
    return trending

def generate_signals(prices):
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    n = len(close)
    
    # Load HTF data ONCE before loop (Rule 1 - CRITICAL)
    df_1d = get_htf_data(prices, '1d')
    df_1w = get_htf_data(prices, '1w')
    
    # Calculate HTF indicators
    hma_1d = calculate_hma(df_1d['close'].values, 21)
    hma_1w = calculate_hma(df_1w['close'].values, 21)
    
    # Align HTF to LTF (Rule 2 - auto shift(1) for completed bars)
    hma_1d_aligned = align_htf_to_ltf(prices, df_1d, hma_1d)
    hma_1w_aligned = align_htf_to_ltf(prices, df_1w, hma_1w)
    
    # Calculate 4h indicators
    atr_14 = calculate_atr(high, low, close, 14)
    atr_7 = calculate_atr(high, low, close, 7)
    atr_21 = calculate_atr(high, low, close, 21)
    fisher = calculate_fisher_transform(high, low, close, 9)
    adx = calculate_adx(high, low, close, 14)
    adx_trending = calculate_adx_hysteresis(adx, 18, 14)
    
    # Volatility regime: ATR(7)/ATR(21) ratio
    vol_ratio = np.full(n, np.nan)
    for i in range(21, n):
        if atr_21[i] > 1e-10:
            vol_ratio[i] = atr_7[i] / atr_21[i]
    
    signals = np.zeros(n)
    
    # Position sizing - discrete levels (Rule 4)
    SIZE_NORMAL = 0.25
    SIZE_HIGH_VOL = 0.15
    
    # Track position state for stoploss
    in_position = False
    position_side = 0
    highest_close = 0.0
    lowest_close = 0.0
    entry_price = 0.0
    entry_atr = 0.0
    
    for i in range(150, n):
        # Skip if indicators not ready
        if np.isnan(atr_14[i]) or atr_14[i] == 0:
            signals[i] = 0.0
            continue
        
        if np.isnan(hma_1d_aligned[i]) or np.isnan(hma_1w_aligned[i]):
            signals[i] = 0.0
            continue
        
        if np.isnan(fisher[i]) or np.isnan(adx[i]):
            signals[i] = 0.0
            continue
        
        if np.isnan(vol_ratio[i]):
            signals[i] = 0.0
            continue
        
        # === DUAL HTF TREND BIAS ===
        # At least ONE of 1d/1w must agree (looser than requiring both)
        bull_1d = close[i] > hma_1d_aligned[i]
        bull_1w = close[i] > hma_1w_aligned[i]
        bear_1d = close[i] < hma_1d_aligned[i]
        bear_1w = close[i] < hma_1w_aligned[i]
        
        bull_bias = bull_1d or bull_1w
        bear_bias = bear_1d or bear_1w
        
        # Strong bias when both agree
        strong_bull = bull_1d and bull_1w
        strong_bear = bear_1d and bear_1w
        
        # === VOLATILITY REGIME ===
        high_vol = vol_ratio[i] > 1.5
        low_vol = vol_ratio[i] < 1.0
        
        # Determine position size based on vol regime
        current_size = SIZE_HIGH_VOL if high_vol else SIZE_NORMAL
        
        # === ADX TRENDING STATE ===
        is_trending = adx_trending[i] == 1
        
        # === ENTRY LOGIC ===
        new_signal = 0.0
        
        # FISHER TRANSFORM REVERSAL SIGNALS (loose thresholds for more trades)
        # Long: Fisher < -1.0 (oversold) + bullish bias
        if fisher[i] < -1.0 and bull_bias:
            new_signal = current_size
        
        # Short: Fisher > +1.0 (overbought) + bearish bias
        elif fisher[i] > 1.0 and bear_bias:
            new_signal = -current_size
        
        # Additional entry: Fisher extreme reversal (< -1.5 or > +1.5) without bias filter
        if fisher[i] < -1.5:
            new_signal = current_size * 0.5  # Half size without HTF confirmation
        elif fisher[i] > 1.5:
            new_signal = -current_size * 0.5
        
        # === STOPLOSS LOGIC (Rule 6) - 2.0 * ATR trailing ===
        if in_position and position_side != 0:
            if position_side > 0:
                if close[i] > highest_close:
                    highest_close = close[i]
                stoploss_price = highest_close - 2.0 * atr_14[i]
                if close[i] < stoploss_price:
                    new_signal = 0.0
            
            if position_side < 0:
                if lowest_close == 0.0 or close[i] < lowest_close:
                    lowest_close = close[i]
                stoploss_price = lowest_close + 2.0 * atr_14[i]
                if close[i] > stoploss_price:
                    new_signal = 0.0
        
        # === TAKE PROFIT - Reduce to half at 2R ===
        if in_position and new_signal != 0.0 and entry_atr > 0:
            if position_side > 0:
                profit = close[i] - entry_price
                if profit >= 2.0 * entry_atr and abs(new_signal) >= SIZE_NORMAL:
                    new_signal = np.sign(new_signal) * current_size * 0.5
            elif position_side < 0:
                profit = entry_price - close[i]
                if profit >= 2.0 * entry_atr and abs(new_signal) >= SIZE_NORMAL:
                    new_signal = np.sign(new_signal) * current_size * 0.5
        
        # === BIAS REVERSAL EXIT ===
        if in_position and new_signal != 0.0:
            if position_side > 0 and strong_bear:
                new_signal = 0.0
            if position_side < 0 and strong_bull:
                new_signal = 0.0
        
        # === UPDATE POSITION TRACKING ===
        if new_signal != 0.0:
            if not in_position:
                in_position = True
                position_side = np.sign(new_signal)
                entry_price = close[i]
                entry_atr = atr_14[i]
                highest_close = close[i] if position_side > 0 else 0.0
                lowest_close = close[i] if position_side < 0 else 0.0
            elif np.sign(new_signal) != position_side:
                position_side = np.sign(new_signal)
                entry_price = close[i]
                entry_atr = atr_14[i]
                highest_close = close[i] if position_side > 0 else 0.0
                lowest_close = close[i] if position_side < 0 else 0.0
        else:
            if in_position:
                in_position = False
                position_side = 0
                entry_price = 0.0
                entry_atr = 0.0
                highest_close = 0.0
                lowest_close = 0.0
        
        signals[i] = new_signal
    
    return signals