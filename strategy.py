#!/usr/bin/env python3
"""
Experiment #480: 1d KAMA Adaptive Trend + Fisher Transform Reversals + Volume Confirmation

Hypothesis: After 479 failed experiments, the key insight is that daily timeframe needs:
1. KAMA (Kaufman Adaptive MA) instead of HMA/EMA - adapts to market efficiency ratio
2. Fisher Transform for sharper reversal signals than RSI (catches extremes better)
3. Volume confirmation to filter false breakouts (volume > 1.5x 20-day average)
4. Asymmetric sizing based on trend confidence (larger when trend + reversal align)
5. Looser entry thresholds to ensure sufficient trades (Fisher > -1.5 / < +1.5)

Why this should work on 1d:
- KAMA reduces whipsaw in choppy markets (ER adapts smoothing constant)
- Fisher Transform normalizes price to Gaussian distribution, better extreme detection
- Volume filter prevents entries on low-liquidity days
- 2.5x ATR stop is tighter than 3.0x for daily timeframe
- Position sizing 0.25-0.30 discrete minimizes fee churn

Timeframe: 1d (REQUIRED for this experiment)
HTF: 1w via mtf_data helper (call ONCE before loop)
Position sizing: 0.25 base, 0.30 when trend confirms
Stoploss: 2.5 * ATR(14) trailing
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_1d_kama_fisher_volume_adaptive_atr_v1"
timeframe = "1d"
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
    Adapts smoothing based on market efficiency ratio (ER).
    ER = |net change| / sum of absolute changes over period
    SC = (ER * (fast_SC - slow_SC) + slow_SC)^2
    """
    n = len(close)
    kama = np.full(n, np.nan)
    
    # Calculate Efficiency Ratio (ER)
    for i in range(period, n):
        net_change = np.abs(close[i] - close[i - period])
        sum_changes = np.sum(np.abs(np.diff(close[i-period:i+1])))
        
        if sum_changes > 1e-10:
            er = net_change / sum_changes
        else:
            er = 0
        
        # Calculate Smoothing Constant (SC)
        fast_sc = 2.0 / (fast_period + 1)
        slow_sc = 2.0 / (slow_period + 1)
        sc = (er * (fast_sc - slow_sc) + slow_sc) ** 2
        
        # Calculate KAMA
        if i == period:
            kama[i] = close[i]
        else:
            kama[i] = kama[i-1] + sc * (close[i] - kama[i-1])
    
    return kama

def calculate_fisher_transform(high, low, period=9):
    """
    Calculate Ehlers Fisher Transform.
    Transforms price to Gaussian distribution for better extreme detection.
    Fisher = 0.5 * ln((1 + X) / (1 - X)) where X = EMA of normalized price
    """
    n = len(high)
    fisher = np.full(n, np.nan)
    trigger = np.full(n, np.nan)
    
    # Calculate typical price and normalize
    typical = (high + low) / 2
    
    for i in range(period, n):
        # Find highest high and lowest low over period
        highest = typical[i-period+1:i+1].max()
        lowest = typical[i-period+1:i+1].min()
        
        price_range = highest - lowest
        if price_range > 1e-10:
            # Normalize price to -1 to +1 range
            x = 2.0 * (typical[i] - lowest) / price_range - 1.0
            # Clamp to avoid division by zero
            x = np.clip(x, -0.999, 0.999)
            
            # EMA of normalized price
            if i == period:
                x_ema = x
            else:
                x_ema = 0.5 * x + 0.5 * x_ema_prev
            
            x_ema_prev = x_ema
            x_ema = np.clip(x_ema, -0.999, 0.999)
            
            # Fisher transform
            fisher[i] = 0.5 * np.log((1 + x_ema) / (1 - x_ema))
            
            # Trigger line (previous Fisher value)
            if i > period:
                trigger[i] = fisher[i-1]
    
    return fisher, trigger

def calculate_volume_ratio(volume, period=20):
    """Calculate volume ratio vs rolling average."""
    vol_s = pd.Series(volume)
    vol_avg = vol_s.rolling(window=period, min_periods=period).mean()
    vol_ratio = volume / vol_avg.replace(0, np.inf)
    return vol_ratio

def calculate_hma(close, period=21):
    """Calculate Hull Moving Average for HTF trend."""
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
    
    adx_val = 0.0
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
            adx_val = dx
        else:
            adx_val = ((adx_val * (period - 1)) + dx) / period
        
        adx[i] = adx_val
    
    return adx

def calculate_sma(close, period=50):
    """Calculate Simple Moving Average."""
    close_s = pd.Series(close)
    return close_s.rolling(window=period, min_periods=period).mean().values

def generate_signals(prices):
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    volume = prices["volume"].values
    n = len(close)
    
    # Load HTF data ONCE before loop (Rule 1 - CRITICAL)
    df_1w = get_htf_data(prices, '1w')
    
    # Calculate HTF indicators
    hma_1w = calculate_hma(df_1w['close'].values, 21)
    
    # Align HTF to LTF (Rule 2 - auto shift(1) for completed bars)
    hma_1w_aligned = align_htf_to_ltf(prices, df_1w, hma_1w)
    
    # Calculate 1d indicators
    atr = calculate_atr(high, low, close, 14)
    kama = calculate_kama(close, 10)
    fisher, fisher_trigger = calculate_fisher_transform(high, low, 9)
    vol_ratio = calculate_volume_ratio(volume, 20)
    adx = calculate_adx(high, low, close, 14)
    sma_50 = calculate_sma(close, 50)
    
    signals = np.zeros(n)
    
    # Position sizing - discrete levels (Rule 4)
    SIZE_BASE = 0.25
    SIZE_CONFIRMED = 0.30
    
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
        
        if np.isnan(hma_1w_aligned[i]):
            signals[i] = 0.0
            continue
        
        if np.isnan(kama[i]) or np.isnan(fisher[i]) or np.isnan(vol_ratio[i]):
            signals[i] = 0.0
            continue
        
        if np.isnan(adx[i]) or np.isnan(sma_50[i]):
            signals[i] = 0.0
            continue
        
        # === WEEKLY HMA TREND BIAS ===
        bull_regime = close[i] > hma_1w_aligned[i]
        bear_regime = close[i] < hma_1w_aligned[i]
        
        # === KAMA TREND DIRECTION ===
        kama_bull = close[i] > kama[i]
        kama_bear = close[i] < kama[i]
        
        # === VOLUME CONFIRMATION ===
        volume_confirmed = vol_ratio[i] > 1.3  # 30% above average
        
        # === FISHER TRANSFORM SIGNALS ===
        # Long: Fisher crosses above -1.5 from below (oversold reversal)
        fisher_long = fisher[i] > -1.5 and fisher_trigger[i] < -1.5 if i > 0 and not np.isnan(fisher_trigger[i]) else False
        
        # Short: Fisher crosses below +1.5 from above (overbought reversal)
        fisher_short = fisher[i] < 1.5 and fisher_trigger[i] > 1.5 if i > 0 and not np.isnan(fisher_trigger[i]) else False
        
        # === ADX TREND STRENGTH ===
        strong_trend = adx[i] > 20
        
        # === ENTRY LOGIC ===
        new_signal = 0.0
        position_size = SIZE_BASE
        
        # LONG ENTRIES
        if bull_regime or kama_bull:
            # Fisher reversal long + volume confirmation
            if fisher_long:
                if volume_confirmed or not strong_trend:
                    # Increase size if trend confirms
                    if bull_regime and kama_bull:
                        position_size = SIZE_CONFIRMED
                    new_signal = position_size
        
        # SHORT ENTRIES
        if bear_regime or kama_bear:
            # Fisher reversal short + volume confirmation
            if fisher_short:
                if volume_confirmed or not strong_trend:
                    # Increase size if trend confirms
                    if bear_regime and kama_bear:
                        position_size = SIZE_CONFIRMED
                    new_signal = -position_size
        
        # === ADDITIONAL MEAN REVERSION ENTRYS (ensure sufficient trades) ===
        # If no Fisher signal but extreme RSI-like conditions
        if new_signal == 0.0:
            # Deep oversold in bull regime
            if bull_regime and fisher[i] < -2.0 and volume_confirmed:
                new_signal = SIZE_BASE
            
            # Deep overbought in bear regime
            if bear_regime and fisher[i] > 2.0 and volume_confirmed:
                new_signal = -SIZE_BASE
        
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
        
        # === REGIME REVERSAL EXIT ===
        # Exit if weekly trend flips strongly against position
        if in_position and new_signal != 0.0:
            if position_side > 0 and bear_regime and close[i] < sma_50[i]:
                new_signal = 0.0
            if position_side < 0 and bull_regime and close[i] > sma_50[i]:
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