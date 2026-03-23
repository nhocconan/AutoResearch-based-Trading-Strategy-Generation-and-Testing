#!/usr/bin/env python3
"""
Experiment #036: 12h Primary + 1d HTF — Fisher Transform + Choppiness Regime + Volume Confirmation

Hypothesis: 12h timeframe with daily trend bias and Fisher Transform entries will generate
25-50 trades/year with positive Sharpe across BTC/ETH/SOL. Key insights from 35 failed experiments:
1) Fisher Transform catches reversals better than RSI in choppy/bear markets (proven on ETH)
2) Volume confirmation on breakouts reduces fake signals (critical for 12h)
3) Choppiness Index regime switching adapts to market conditions
4) 1d HTF trend bias filters counter-trend trades (improves win rate)
5) Asymmetric position sizing based on regime volatility

Strategy Logic:
1. FISHER TRANSFORM (period=9): Entry timing - long when crosses above -1.5, short when crosses below +1.5
2. CHOPPINESS INDEX (period=14): Regime detection - CHOP>55=range, CHOP<45=trend
3. VOLUME SPIKE: Volume > 1.5*MA(20) confirms breakouts (avoids fake moves)
4. 1d HMA(21): Macro trend bias from HTF
5. ATR(14) trailing stoploss: 2.5*ATR for trend regime, 2.0*ATR for range regime

Why this should work:
- 12h primary = fewer trades than 4h, less fee drag (targets 25-50/year)
- 1d HTF = strong trend filter without being too slow (like 1w)
- Fisher Transform = better reversal detection than RSI in bear/range markets
- Volume confirmation = filters fake breakouts (major issue on 12h)
- Regime-adaptive stops = tighter in range, wider in trend

Position size: 0.25 (discrete, conservative for 12h)
Stoploss: 2.0-2.5*ATR trailing based on regime
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_12h_fisher_chop_volume_regime_1d_v1"
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

def calculate_hma(close, period=21):
    """Calculate Hull Moving Average (HMA)."""
    close_s = pd.Series(close)
    
    half = int(period / 2)
    sqrt_n = int(np.sqrt(period))
    
    wma_half = close_s.ewm(span=half, min_periods=half, adjust=False).mean()
    wma_full = close_s.ewm(span=period, min_periods=period, adjust=False).mean()
    
    raw_hma = 2.0 * wma_half - wma_full
    hma = raw_hma.ewm(span=sqrt_n, min_periods=sqrt_n, adjust=False).mean()
    
    return hma.values

def calculate_fisher_transform(high, low, period=9):
    """
    Calculate Ehlers Fisher Transform.
    Converts price into a Gaussian normal distribution for clearer reversal signals.
    Entry: Fisher crosses above -1.5 (long), crosses below +1.5 (short)
    """
    hl2 = (high + low) / 2.0
    hl2_s = pd.Series(hl2)
    
    # Normalize price to range -1 to +1
    highest = hl2_s.rolling(window=period, min_periods=period).max().values
    lowest = hl2_s.rolling(window=period, min_periods=period).min().values
    price_range = highest - lowest + 1e-10
    
    normalized = 2.0 * (hl2 - lowest) / price_range - 1.0
    normalized = np.clip(normalized, -0.999, 0.999)  # Prevent log domain errors
    
    # Fisher transform
    fisher = 0.5 * np.log((1.0 + normalized) / (1.0 - normalized + 1e-10))
    fisher_s = pd.Series(fisher)
    fisher_ema = fisher_s.ewm(span=period, min_periods=period, adjust=False).mean().values
    
    return fisher_ema

def calculate_choppiness(high, low, close, period=14):
    """
    Calculate Choppiness Index (CHOP).
    CHOP = 100 * LOG10(SUM(ATR, n) / (Highest High - Lowest Low)) / LOG10(n)
    """
    n = period
    
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]
    
    atr_sum = pd.Series(tr).rolling(window=n, min_periods=n).sum().values
    highest_high = pd.Series(high).rolling(window=n, min_periods=n).max().values
    lowest_low = pd.Series(low).rolling(window=n, min_periods=n).min().values
    
    price_range = highest_high - lowest_low + 1e-10
    chop = 100.0 * np.log10(atr_sum / price_range) / np.log10(n)
    
    return chop

def calculate_volume_spike(volume, period=20, threshold=1.5):
    """Detect volume spikes above moving average."""
    vol_ma = pd.Series(volume).rolling(window=period, min_periods=period).mean().values
    is_spike = volume > (threshold * vol_ma)
    return is_spike

def calculate_kama(close, er_period=10, fast_period=2, slow_period=30):
    """
    Calculate Kaufman Adaptive Moving Average (KAMA).
    Adapts to market noise - moves fast in trends, slow in ranges.
    """
    close_s = pd.Series(close)
    n = len(close)
    
    # Efficiency Ratio
    noise = np.abs(close_s.diff()).rolling(window=er_period, min_periods=er_period).sum().values
    signal = np.abs(close_s.diff(er_period)).values
    signal[:er_period] = np.nan
    er = signal / (noise + 1e-10)
    er = np.nan_to_num(er, nan=0.0)
    
    # Smoothing constant
    fast_sc = 2.0 / (fast_period + 1.0)
    slow_sc = 2.0 / (slow_period + 1.0)
    sc = (er * (fast_sc - slow_sc) + slow_sc) ** 2
    
    # KAMA calculation
    kama = np.zeros(n)
    kama[0] = close[0]
    for i in range(1, n):
        if np.isnan(sc[i]):
            kama[i] = kama[i-1]
        else:
            kama[i] = kama[i-1] + sc[i] * (close[i] - kama[i-1])
    
    return kama

def generate_signals(prices):
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    volume = prices["volume"].values
    n = len(close)
    
    # Load HTF data ONCE before loop (Rule 1 - CRITICAL)
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate 1d HMA for macro bias
    hma_1d = calculate_hma(df_1d['close'].values, period=21)
    hma_1d_aligned = align_htf_to_ltf(prices, df_1d, hma_1d)
    
    # Calculate 12h indicators
    atr_14 = calculate_atr(high, low, close, period=14)
    fisher_9 = calculate_fisher_transform(high, low, period=9)
    chop_14 = calculate_choppiness(high, low, close, period=14)
    kama_10 = calculate_kama(close, er_period=10)
    vol_spike = calculate_volume_spike(volume, period=20, threshold=1.5)
    
    # Calculate HMA for trend confirmation
    hma_21 = calculate_hma(close, period=21)
    hma_48 = calculate_hma(close, period=48)
    
    # Calculate Bollinger Bands for range detection
    bb_period = 20
    bb_std = 2.0
    close_s = pd.Series(close)
    bb_mid = close_s.rolling(window=bb_period, min_periods=bb_period).mean().values
    bb_std_val = close_s.rolling(window=bb_period, min_periods=bb_period).std().values
    bb_upper = bb_mid + bb_std * bb_std_val
    bb_lower = bb_mid - bb_std * bb_std_val
    
    signals = np.zeros(n)
    
    # Position sizing (Rule 4 - discrete, max 0.40)
    POSITION_SIZE = 0.25  # Conservative for 12h
    
    # Track position state for stoploss
    in_position = False
    position_side = 0
    highest_since_entry = 0.0
    lowest_since_entry = 0.0
    entry_price = 0.0
    entry_fisher = 0.0
    
    for i in range(60, n):
        # Skip if indicators not ready
        if np.isnan(hma_1d_aligned[i]) or np.isnan(atr_14[i]):
            continue
        if np.isnan(fisher_9[i]) or np.isnan(chop_14[i]) or np.isnan(kama_10[i]):
            continue
        if np.isnan(bb_upper[i]) or np.isnan(hma_21[i]) or atr_14[i] == 0:
            continue
        
        # === 1D MACRO BIAS ===
        price_above_hma_1d = close[i] > hma_1d_aligned[i]
        price_below_hma_1d = close[i] < hma_1d_aligned[i]
        
        # === CHOPPINESS REGIME ===
        chop_value = chop_14[i]
        is_ranging = chop_value > 55.0  # Range market
        is_trending = chop_value < 45.0  # Trend market (with hysteresis)
        
        # === FISHER TRANSFORM SIGNALS ===
        fisher_current = fisher_9[i]
        fisher_prev = fisher_9[i-1] if i > 0 else fisher_current
        
        fisher_long_signal = (fisher_prev <= -1.5 and fisher_current > -1.5)
        fisher_short_signal = (fisher_prev >= 1.5 and fisher_current < 1.5)
        fisher_extreme_low = fisher_current < -2.0
        fisher_extreme_high = fisher_current > 2.0
        
        # === KAMA TREND ===
        kama_bullish = close[i] > kama_10[i]
        kama_bearish = close[i] < kama_10[i]
        kama_slope_up = kama_10[i] > kama_10[i-3] if i > 3 else False
        kama_slope_down = kama_10[i] < kama_10[i-3] if i > 3 else False
        
        # === HMA TREND ===
        hma_bullish = close[i] > hma_21[i]
        hma_bearish = close[i] < hma_21[i]
        hma_cross_bullish = (close[i-1] <= hma_21[i-1] and close[i] > hma_21[i])
        hma_cross_bearish = (close[i-1] >= hma_21[i-1] and close[i] < hma_21[i])
        
        # === BOLLINGER BAND POSITION ===
        price_below_bb_lower = close[i] < bb_lower[i]
        price_above_bb_upper = close[i] > bb_upper[i]
        price_near_bb_lower = close[i] < bb_lower[i] * 1.01
        price_near_bb_upper = close[i] > bb_upper[i] * 0.99
        
        # === VOLUME CONFIRMATION ===
        volume_confirmed = vol_spike[i]
        
        # === ADAPTIVE REGIME ENTRY LOGIC ===
        new_signal = 0.0
        
        # --- RANGING REGIME: Mean Reversion with Fisher ---
        if is_ranging:
            # Long: Fisher extreme low + price near BB lower + daily bias helps
            if fisher_extreme_low or (fisher_long_signal and price_near_bb_lower):
                if price_above_hma_1d or kama_bullish:  # Daily bullish OR KAMA confirms
                    new_signal = POSITION_SIZE
            
            # Short: Fisher extreme high + price near BB upper + daily bias helps
            elif fisher_extreme_high or (fisher_short_signal and price_near_bb_upper):
                if price_below_hma_1d or kama_bearish:  # Daily bearish OR KAMA confirms
                    new_signal = -POSITION_SIZE
        
        # --- TRENDING REGIME: Trend Following with Volume ---
        elif is_trending:
            # Long: HMA cross bullish + KAMA confirms + volume spike + daily bias
            if hma_cross_bullish and kama_bullish:
                if volume_confirmed and price_above_hma_1d:
                    new_signal = POSITION_SIZE
            # Also allow Fisher reversal in trend pullbacks
            elif fisher_long_signal and kama_bullish:
                if price_above_hma_1d and close[i] > hma_48[i]:
                    new_signal = POSITION_SIZE
            
            # Short: HMA cross bearish + KAMA confirms + volume spike + daily bias
            elif hma_cross_bearish and kama_bearish:
                if volume_confirmed and price_below_hma_1d:
                    new_signal = -POSITION_SIZE
            # Also allow Fisher reversal in trend pullbacks
            elif fisher_short_signal and kama_bearish:
                if price_below_hma_1d and close[i] < hma_48[i]:
                    new_signal = -POSITION_SIZE
        
        # --- NEUTRAL REGIME: HMA + KAMA confluence ---
        else:
            # Long: Both HMA and KAMA bullish + daily bias
            if hma_bullish and kama_bullish and kama_slope_up:
                if price_above_hma_1d:
                    new_signal = POSITION_SIZE
            
            # Short: Both HMA and KAMA bearish + daily bias
            elif hma_bearish and kama_bearish and kama_slope_down:
                if price_below_hma_1d:
                    new_signal = -POSITION_SIZE
        
        # === HOLD POSITION LOGIC ===
        if in_position and new_signal == 0.0:
            new_signal = signals[i-1] if i > 0 else 0.0
        
        # === STOPLOSS CHECK (Regime-adaptive ATR trailing) ===
        stoploss_triggered = False
        stop_multiplier = 2.0 if is_ranging else 2.5  # Tighter in range, wider in trend
        
        if in_position and position_side > 0:
            highest_since_entry = max(highest_since_entry, close[i])
            stop_price = highest_since_entry - stop_multiplier * atr_14[i]
            if close[i] < stop_price:
                stoploss_triggered = True
        
        if in_position and position_side < 0:
            if lowest_since_entry == 0.0:
                lowest_since_entry = close[i]
            else:
                lowest_since_entry = min(lowest_since_entry, close[i])
            stop_price = lowest_since_entry + stop_multiplier * atr_14[i]
            if close[i] > stop_price:
                stoploss_triggered = True
        
        if stoploss_triggered:
            new_signal = 0.0
        
        # === EXIT ON REGIME/TREND CHANGE ===
        # Exit long if daily trend turns strongly bearish
        if in_position and position_side > 0:
            if price_below_hma_1d and kama_bearish and chop_value < 40:
                new_signal = 0.0
        
        # Exit short if daily trend turns strongly bullish
        if in_position and position_side < 0:
            if price_above_hma_1d and kama_bullish and chop_value < 40:
                new_signal = 0.0
        
        # === UPDATE POSITION TRACKING ===
        if new_signal != 0.0:
            if not in_position:
                in_position = True
                position_side = np.sign(new_signal)
                entry_price = close[i]
                highest_since_entry = close[i] if position_side > 0 else 0.0
                lowest_since_entry = close[i] if position_side < 0 else 0.0
                entry_fisher = fisher_9[i]
            elif np.sign(new_signal) != position_side:
                position_side = np.sign(new_signal)
                entry_price = close[i]
                highest_since_entry = close[i] if position_side > 0 else 0.0
                lowest_since_entry = close[i] if position_side < 0 else 0.0
                entry_fisher = fisher_9[i]
        else:
            if in_position:
                in_position = False
                position_side = 0
                entry_price = 0.0
                highest_since_entry = 0.0
                lowest_since_entry = 0.0
                entry_fisher = 0.0
        
        signals[i] = new_signal
    
    return signals