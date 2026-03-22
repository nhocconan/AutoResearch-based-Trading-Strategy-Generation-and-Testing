#!/usr/bin/env python3
"""
Experiment #501: 1h Fisher Transform + 4h HMA Trend + ADX Hysteresis

Hypothesis: After 500 failed experiments, the key insight is that 1h timeframe needs:
1. Fisher Transform (Ehlers) for reversal detection - works well in bear/range markets
2. 4h HMA(21) for robust trend bias without whipsaw
3. ADX hysteresis (25 enter, 18 exit) to prevent signal churn
4. Volume confirmation to filter false breakouts
5. Asymmetric logic: long on dips in bull, short on rallies in bear

Why this should work on 1h:
- Fisher Transform normalizes price to -1.5 to +1.5 range, catching reversals
- 4h HMA provides stable trend bias (proven in successful strategies)
- ADX hysteresis reduces trade frequency but improves quality
- Volume ratio > 1.2 confirms genuine moves
- Should generate 30-50 trades/year per symbol (enough for Sharpe)

Timeframe: 1h (REQUIRED for this experiment)
HTF: 4h via mtf_data helper (call ONCE before loop)
Position sizing: 0.25-0.30 discrete levels
Stoploss: 2.5 * ATR(14) trailing
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_1h_fisher_4h_hma_adx_hysteresis_volume_atr_v1"
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

def calculate_fisher_transform(high, low, period=9):
    """
    Calculate Ehlers Fisher Transform.
    Normalizes price to -1.5 to +1.5 range for reversal detection.
    Long when Fisher crosses above -1.5, short when crosses below +1.5.
    """
    n = len(high)
    fisher = np.full(n, np.nan)
    trigger = np.full(n, np.nan)
    
    for i in range(period, n):
        # Calculate typical price
        hl2 = (high[i-period+1:i+1].max() + low[i-period+1:i+1].min()) / 2.0
        
        # Normalize to -1 to +1 range
        highest = high[i-period+1:i+1].max()
        lowest = low[i-period+1:i+1].min()
        
        if highest - lowest > 1e-10:
            normalized = 2.0 * (hl2 - lowest) / (highest - lowest) - 1.0
            normalized = max(-0.999, min(0.999, normalized))  # Clamp to avoid log errors
            
            # Fisher transform
            fisher_val = 0.5 * np.log((1 + normalized) / (1 - normalized))
            
            # Smooth with EMA
            if i == period:
                fisher[i] = fisher_val
            else:
                fisher[i] = 0.67 * fisher[i-1] + 0.33 * fisher_val
            
            # Trigger line (1-period lag)
            trigger[i] = fisher[i-1] if i > period else fisher_val
    
    return fisher, trigger

def calculate_volume_ratio(volume, period=20):
    """Calculate volume ratio relative to rolling average."""
    vol_s = pd.Series(volume)
    vol_avg = vol_s.rolling(window=period, min_periods=period).mean()
    vol_ratio = vol_s / vol_avg.replace(0, np.inf)
    return vol_ratio.values

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

def calculate_sma(close, period=200):
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
    df_4h = get_htf_data(prices, '4h')
    
    # Calculate HTF indicators
    hma_4h = calculate_hma(df_4h['close'].values, 21)
    
    # Align HTF to LTF (Rule 2 - auto shift(1) for completed bars)
    hma_4h_aligned = align_htf_to_ltf(prices, df_4h, hma_4h)
    
    # Calculate 1h indicators
    atr = calculate_atr(high, low, close, 14)
    adx = calculate_adx(high, low, close, 14)
    fisher, fisher_trigger = calculate_fisher_transform(high, low, 9)
    vol_ratio = calculate_volume_ratio(volume, 20)
    rsi = calculate_rsi(close, 14)
    sma_200 = calculate_sma(close, 200)
    
    signals = np.zeros(n)
    
    # Position sizing - discrete levels (Rule 4)
    SIZE_BASE = 0.25
    SIZE_MAX = 0.30
    
    # Track position state for stoploss and ADX hysteresis
    in_position = False
    position_side = 0
    highest_close = 0.0
    lowest_close = 0.0
    entry_price = 0.0
    adx_active = False  # ADX hysteresis state
    
    for i in range(100, n):
        # Skip if indicators not ready
        if np.isnan(atr[i]) or atr[i] == 0:
            signals[i] = 0.0
            continue
        
        if np.isnan(hma_4h_aligned[i]):
            signals[i] = 0.0
            continue
        
        if np.isnan(fisher[i]) or np.isnan(fisher_trigger[i]):
            signals[i] = 0.0
            continue
        
        if np.isnan(adx[i]) or np.isnan(vol_ratio[i]) or np.isnan(rsi[i]):
            signals[i] = 0.0
            continue
        
        if np.isnan(sma_200[i]):
            signals[i] = 0.0
            continue
        
        # === 4H HMA TREND BIAS ===
        bull_regime = close[i] > hma_4h_aligned[i]
        bear_regime = close[i] < hma_4h_aligned[i]
        
        # === ADX HYSTERESIS (25 enter, 18 exit) ===
        if not adx_active and adx[i] > 25:
            adx_active = True
        elif adx_active and adx[i] < 18:
            adx_active = False
        
        # === VOLUME CONFIRMATION ===
        volume_confirmed = vol_ratio[i] > 1.2
        
        # === FISHER TRANSFORM SIGNALS ===
        fisher_long = fisher[i] > -1.5 and fisher_trigger[i] < -1.5
        fisher_short = fisher[i] < 1.5 and fisher_trigger[i] > 1.5
        
        # Fisher crossover signals
        fisher_cross_up = fisher[i] > fisher_trigger[i] and fisher[i-1] <= fisher_trigger[i-1] if i > 0 else False
        fisher_cross_down = fisher[i] < fisher_trigger[i] and fisher[i-1] >= fisher_trigger[i-1] if i > 0 else False
        
        # === ASYMMETRIC ENTRY LOGIC ===
        new_signal = 0.0
        
        # BULL REGIME: Favor long entries on dips
        if bull_regime:
            # Primary: Fisher cross up + volume confirmation
            if fisher_cross_up and volume_confirmed:
                new_signal = SIZE_BASE
            
            # Secondary: RSI oversold + above SMA200
            elif rsi[i] < 35 and close[i] > sma_200[i]:
                new_signal = SIZE_BASE
            
            # Tertiary: Fisher extreme oversold
            elif fisher[i] < -1.8:
                new_signal = SIZE_BASE * 0.8
        
        # BEAR REGIME: Favor short entries on rallies
        if bear_regime:
            # Primary: Fisher cross down + volume confirmation
            if fisher_cross_down and volume_confirmed:
                new_signal = -SIZE_BASE
            
            # Secondary: RSI overbought + below SMA200
            elif rsi[i] > 65 and close[i] < sma_200[i]:
                new_signal = -SIZE_BASE
            
            # Tertiary: Fisher extreme overbought
            elif fisher[i] > 1.8:
                new_signal = -SIZE_BASE * 0.8
        
        # === ADX FILTER (only allow trades when ADX active) ===
        # But allow mean-reversion trades even when ADX low
        if adx_active:
            # Trending market - allow full size
            pass
        else:
            # Ranging market - reduce size for mean-reversion only
            if new_signal != 0.0:
                # Only allow if Fisher is at extreme (mean-reversion)
                if abs(fisher[i]) < 1.5:
                    new_signal = 0.0
                else:
                    new_signal = new_signal * 0.7  # Reduce size in range
        
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
        # Exit if 4h trend flips against position
        if in_position and new_signal != 0.0:
            if position_side > 0 and bear_regime:
                new_signal = 0.0
            if position_side < 0 and bull_regime:
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