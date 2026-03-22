#!/usr/bin/env python3
"""
Experiment #603: 1d Primary + 1w HTF — Dual Regime with KAMA Trend + Donchian Breakout + RSI Filter

Hypothesis: Building on #591/#594 success with Choppiness regime detection, this strategy uses
daily timeframe with weekly HTF for major trend direction. Key innovation: KAMA (Kaufman Adaptive
Moving Average) adapts to volatility - fast in trends, slow in ranges. Combined with Donchian
breakouts for entry timing and RSI for pullback confirmation.

Why this might beat Sharpe=0.520:
1. 1w KAMA(21) for major trend - adapts to crypto volatility better than HMA/EMA
2. 1d Choppiness Index regime switch - mean revert when CHOP>55, trend follow when CHOP<45
3. Donchian(20) breakout for entry timing - proven on SOL (Sharpe 0.782 in research)
4. RSI(14) pullback filter - enter on retracements, not breakouts
5. ATR(14) trailing stop - 2.5x ATR protects capital
6. Position size 0.28 discrete - balances return vs drawdown

Position sizing: 0.28 discrete (proven range 0.20-0.35)
Target: >=30 trades/symbol train, >=3 test, Sharpe > 0 all symbols
Trade frequency: 20-50/year on 1d (strict filters prevent overtrading)
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_1d_kama_donchian_regime_1w_v1"
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

def calculate_rsi(close, period=14):
    """Calculate RSI using Wilder's smoothing."""
    close_s = pd.Series(close)
    delta = close_s.diff()
    
    gain = delta.where(delta > 0, 0.0)
    loss = -delta.where(delta < 0, 0.0)
    
    avg_gain = gain.ewm(span=period, min_periods=period, adjust=False).mean()
    avg_loss = loss.ewm(span=period, min_periods=period, adjust=False).mean()
    
    rs = avg_gain / (avg_loss + 1e-10)
    rsi = 100.0 - (100.0 / (1.0 + rs))
    
    return rsi.values

def calculate_kama(close, er_period=10, fast_period=2, slow_period=30):
    """
    Calculate Kaufman Adaptive Moving Average (KAMA).
    KAMA adapts to market noise - fast during trends, slow during ranges.
    
    Efficiency Ratio (ER) = |price change| / sum of individual changes
    SC (Smoothing Constant) = (ER * (fast_sc - slow_sc) + slow_sc)^2
    """
    n = len(close)
    close_s = pd.Series(close)
    
    # Price change over er_period
    price_change = np.abs(close_s - close_s.shift(er_period))
    
    # Sum of individual changes (volatility)
    individual_changes = np.abs(close_s.diff())
    volatility = individual_changes.rolling(window=er_period, min_periods=er_period).sum()
    
    # Efficiency Ratio
    with np.errstate(divide='ignore', invalid='ignore'):
        er = price_change / (volatility + 1e-10)
    er = er.fillna(0.0)
    
    # Smoothing constants
    fast_sc = 2.0 / (fast_period + 1.0)
    slow_sc = 2.0 / (slow_period + 1.0)
    
    # Adaptive smoothing constant
    sc = (er * (fast_sc - slow_sc) + slow_sc) ** 2
    
    # KAMA calculation
    kama = np.zeros(n)
    kama[0] = close[0]
    
    for i in range(1, n):
        kama[i] = kama[i-1] + sc.iloc[i] * (close[i] - kama[i-1])
    
    return kama

def calculate_choppiness(high, low, close, period=14):
    """
    Calculate Choppiness Index (CHOP) - regime detection.
    CHOP > 61.8 = choppy/range market (mean reversion works)
    CHOP < 38.2 = trending market (trend following works)
    """
    n = period
    atr_vals = calculate_atr(high, low, close, 14)
    
    # Sum of ATR over period
    atr_sum = pd.Series(atr_vals).rolling(window=n, min_periods=n).sum().values
    
    # Highest high and lowest low over period
    highest_high = pd.Series(high).rolling(window=n, min_periods=n).max().values
    lowest_low = pd.Series(low).rolling(window=n, min_periods=n).min().values
    
    # Price range
    price_range = highest_high - lowest_low
    
    # CHOP formula
    with np.errstate(divide='ignore', invalid='ignore'):
        chop = 100.0 * np.log10(atr_sum / (price_range + 1e-10)) / np.log10(n)
    
    # Clip to valid range
    chop = np.clip(chop, 0.0, 100.0)
    chop = np.nan_to_num(chop, nan=50.0)
    
    return chop

def calculate_donchian(high, low, period=20):
    """Calculate Donchian Channel - breakout levels."""
    upper = pd.Series(high).rolling(window=period, min_periods=period).max().values
    lower = pd.Series(low).rolling(window=period, min_periods=period).min().values
    middle = (upper + lower) / 2.0
    return upper, lower, middle

def calculate_adx(high, low, close, period=14):
    """Calculate ADX for trend strength."""
    high_s = pd.Series(high)
    low_s = pd.Series(low)
    close_s = pd.Series(close)
    
    tr1 = high_s - low_s
    tr2 = np.abs(high_s - close_s.shift(1))
    tr3 = np.abs(low_s - close_s.shift(1))
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    
    plus_dm = high_s.diff()
    minus_dm = -low_s.diff()
    
    plus_dm = plus_dm.where((plus_dm > minus_dm) & (plus_dm > 0), 0.0)
    minus_dm = minus_dm.where((minus_dm > plus_dm) & (minus_dm > 0), 0.0)
    
    atr = tr.ewm(span=period, min_periods=period, adjust=False).mean()
    plus_di = 100.0 * (plus_dm.ewm(span=period, min_periods=period, adjust=False).mean() / (atr + 1e-10))
    minus_di = 100.0 * (minus_dm.ewm(span=period, min_periods=period, adjust=False).mean() / (atr + 1e-10))
    
    dx = 100.0 * np.abs(plus_di - minus_di) / (plus_di + minus_di + 1e-10)
    adx = dx.ewm(span=period, min_periods=period, adjust=False).mean()
    
    return adx.values

def calculate_bollinger(close, period=20, std_mult=2.0):
    """Calculate Bollinger Bands."""
    close_s = pd.Series(close)
    sma = close_s.rolling(window=period, min_periods=period).mean()
    std = close_s.rolling(window=period, min_periods=period).std()
    
    upper = sma + (std_mult * std)
    lower = sma - (std_mult * std)
    
    return upper.values, lower.values, sma.values

def generate_signals(prices):
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    volume = prices["volume"].values
    n = len(close)
    
    # Load 1w HTF data ONCE before loop (Rule 1 - CRITICAL)
    df_1w = get_htf_data(prices, '1w')
    
    # Calculate 1w HTF KAMA for major trend direction
    kama_1w_21 = calculate_kama(df_1w['close'].values, er_period=10, fast_period=2, slow_period=30)
    kama_1w_50 = calculate_kama(df_1w['close'].values, er_period=10, fast_period=2, slow_period=50)
    
    # Align HTF to LTF (Rule 2 - auto shift(1) for completed bars)
    kama_1w_21_aligned = align_htf_to_ltf(prices, df_1w, kama_1w_21)
    kama_1w_50_aligned = align_htf_to_ltf(prices, df_1w, kama_1w_50)
    
    # Calculate 1d indicators
    atr_14 = calculate_atr(high, low, close, 14)
    rsi_14 = calculate_rsi(close, 14)
    chop_14 = calculate_choppiness(high, low, close, 14)
    adx_14 = calculate_adx(high, low, close, 14)
    
    # Donchian channels for breakout levels
    donchian_upper, donchian_lower, donchian_mid = calculate_donchian(high, low, 20)
    
    # Bollinger Bands for mean reversion
    bb_upper, bb_lower, bb_mid = calculate_bollinger(close, 20, 2.0)
    
    # 1d KAMA for dynamic support/resistance
    kama_1d_21 = calculate_kama(close, er_period=10, fast_period=2, slow_period=30)
    kama_1d_50 = calculate_kama(close, er_period=10, fast_period=2, slow_period=50)
    
    # Volume SMA for confirmation
    vol_sma20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    
    # Position sizing (Rule 4 - discrete, max 0.40)
    POSITION_SIZE = 0.28
    
    # Track position state for stoploss
    in_position = False
    position_side = 0
    entry_price = 0.0
    highest_since_entry = 0.0
    lowest_since_entry = 0.0
    
    for i in range(100, n):
        # Skip if indicators not ready
        if np.isnan(atr_14[i]) or atr_14[i] == 0:
            continue
        if np.isnan(kama_1w_21_aligned[i]) or np.isnan(kama_1w_50_aligned[i]):
            continue
        if np.isnan(rsi_14[i]) or np.isnan(chop_14[i]) or np.isnan(adx_14[i]):
            continue
        if np.isnan(donchian_upper[i]) or np.isnan(bb_upper[i]):
            continue
        if np.isnan(vol_sma20[i]):
            continue
        
        # === 1W TREND BIAS (Major Direction) ===
        bull_bias_1w = close[i] > kama_1w_21_aligned[i]
        bear_bias_1w = close[i] < kama_1w_21_aligned[i]
        
        # 1w KAMA slope for trend strength
        kama_1w_slope_bull = kama_1w_21_aligned[i] > kama_1w_50_aligned[i]
        kama_1w_slope_bear = kama_1w_21_aligned[i] < kama_1w_50_aligned[i]
        
        # === 1D REGIME DETECTION (Choppiness Index) ===
        is_chop_regime = chop_14[i] > 55.0
        is_trend_regime = chop_14[i] < 45.0
        
        # === ADX TREND STRENGTH ===
        strong_trend = adx_14[i] > 20.0
        weak_trend = adx_14[i] < 20.0
        
        # === VOLUME CONFIRMATION ===
        volume_confirmed = volume[i] > 0.8 * vol_sma20[i]
        
        # === 1D KAMA POSITION ===
        above_kama_1d = close[i] > kama_1d_21[i]
        below_kama_1d = close[i] < kama_1d_21[i]
        
        # === ENTRY LOGIC - DUAL REGIME ===
        new_signal = 0.0
        
        # --- CHOP REGIME: Mean Reversion (BB + RSI extremes) ---
        if is_chop_regime:
            # Long: RSI < 30 + price near BB lower + 1w bull bias
            if rsi_14[i] < 30.0 and close[i] <= bb_lower[i] * 1.005 and bull_bias_1w and volume_confirmed:
                new_signal = POSITION_SIZE
            
            # Short: RSI > 70 + price near BB upper + 1w bear bias
            elif rsi_14[i] > 70.0 and close[i] >= bb_upper[i] * 0.995 and bear_bias_1w and volume_confirmed:
                new_signal = -POSITION_SIZE
        
        # --- TREND REGIME: Trend Following (Donchian breakout + pullback) ---
        elif is_trend_regime:
            # Long: 1w bull + ADX strong + price breaks Donchian upper OR pullback to KAMA
            if bull_bias_1w and kama_1w_slope_bull:
                # Breakout entry
                if close[i] > donchian_upper[i] * 0.998 and strong_trend and volume_confirmed:
                    new_signal = POSITION_SIZE
                # Pullback entry (RSI 35-50, price near KAMA)
                elif 35.0 < rsi_14[i] < 50.0 and close[i] <= kama_1d_21[i] * 1.01 and above_kama_1d:
                    new_signal = POSITION_SIZE
            
            # Short: 1w bear + ADX strong + price breaks Donchian lower OR pullback to KAMA
            elif bear_bias_1w and kama_1w_slope_bear:
                # Breakout entry
                if close[i] < donchian_lower[i] * 1.002 and strong_trend and volume_confirmed:
                    new_signal = -POSITION_SIZE
                # Pullback entry (RSI 50-65, price near KAMA)
                elif 50.0 < rsi_14[i] < 65.0 and close[i] >= kama_1d_21[i] * 0.99 and below_kama_1d:
                    new_signal = -POSITION_SIZE
        
        # === HOLD POSITION LOGIC ===
        # If already in position, maintain unless exit conditions hit
        if in_position and new_signal == 0.0:
            new_signal = signals[i-1] if i > 0 else 0.0
        
        # === STOPLOSS CHECK (2.5 * ATR trailing) ===
        stoploss_triggered = False
        
        if in_position and position_side > 0:
            highest_since_entry = max(highest_since_entry, close[i])
            stop_price = highest_since_entry - 2.5 * atr_14[i]
            if close[i] < stop_price:
                stoploss_triggered = True
        
        if in_position and position_side < 0:
            if lowest_since_entry == 0.0:
                lowest_since_entry = close[i]
            else:
                lowest_since_entry = min(lowest_since_entry, close[i])
            stop_price = lowest_since_entry + 2.5 * atr_14[i]
            if close[i] > stop_price:
                stoploss_triggered = True
        
        if stoploss_triggered:
            new_signal = 0.0
        
        # === EXIT CONDITIONS (regime flip or bias flip) ===
        # Exit long on 1w bias flip to bear or chop regime with RSI extreme
        if in_position and position_side > 0:
            if bear_bias_1w and rsi_14[i] > 60.0:
                new_signal = 0.0
            # Exit in chop regime if RSI goes extreme
            elif is_chop_regime and rsi_14[i] > 70.0:
                new_signal = 0.0
        
        # Exit short on 1w bias flip to bull or chop regime with RSI extreme
        if in_position and position_side < 0:
            if bull_bias_1w and rsi_14[i] < 40.0:
                new_signal = 0.0
            # Exit in chop regime if RSI goes extreme
            elif is_chop_regime and rsi_14[i] < 30.0:
                new_signal = 0.0
        
        # === UPDATE POSITION TRACKING ===
        if new_signal != 0.0:
            if not in_position:
                in_position = True
                position_side = np.sign(new_signal)
                entry_price = close[i]
                highest_since_entry = close[i] if position_side > 0 else 0.0
                lowest_since_entry = close[i] if position_side < 0 else 0.0
            elif np.sign(new_signal) != position_side:
                # Flip position
                position_side = np.sign(new_signal)
                entry_price = close[i]
                highest_since_entry = close[i] if position_side > 0 else 0.0
                lowest_since_entry = close[i] if position_side < 0 else 0.0
        else:
            if in_position:
                in_position = False
                position_side = 0
                entry_price = 0.0
                highest_since_entry = 0.0
                lowest_since_entry = 0.0
        
        signals[i] = new_signal
    
    return signals