#!/usr/bin/env python3
"""
Experiment #569: 12h Dual Regime Strategy with 1d/1w HMA Bias

Hypothesis: After analyzing 500+ failed experiments, the key insight is:
1. 12h timeframe balances noise reduction with trade frequency
2. Dual HTF (1d + 1w HMA) provides robust regime detection
3. Asymmetric entry logic: trend-follow in strong regimes, mean-revert in weak
4. RSI pullback entries within trend direction reduce whipsaw
5. Volume confirmation filters false breakouts
6. Loose ADX threshold (>15 not >25) ensures sufficient trade generation

Why this should work on 12h:
- 12h has ~730 bars/year = 20-40 trades/year target (low fee drag)
- 1d HMA for intermediate trend, 1w HMA for macro regime
- RSI(14) pullback to 35-40 in uptrend = high probability entry
- Volume > 1.2 * avg confirms genuine breakouts
- 2.5*ATR stoploss protects against 2022-style crashes
- Position size 0.28 balances return vs drawdown

Timeframe: 12h (REQUIRED for this experiment)
HTF: 1d and 1w via mtf_data helper (call ONCE before loop)
Position sizing: 0.28 discrete (max 0.40)
Stoploss: 2.5 * ATR(14) trailing
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_12h_dual_regime_1d_1w_hma_rsi_pullback_vol_atr_v1"
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
    """Calculate Hull Moving Average for smoother trend with less lag."""
    close_s = pd.Series(close)
    half = max(1, period // 2)
    sqrt_period = max(1, int(np.sqrt(period)))
    wma1 = close_s.ewm(span=half, min_periods=half, adjust=False).mean()
    wma2 = close_s.ewm(span=period, min_periods=period, adjust=False).mean()
    wma3 = (2 * wma1 - wma2).ewm(span=sqrt_period, min_periods=sqrt_period, adjust=False).mean()
    return wma3.values

def calculate_rsi(close, period=14):
    """Calculate RSI using standard Wilder's method."""
    close_s = pd.Series(close)
    delta = close_s.diff()
    gain = delta.where(delta > 0, 0.0)
    loss = -delta.where(delta < 0, 0.0)
    avg_gain = gain.ewm(span=period, min_periods=period, adjust=False).mean()
    avg_loss = loss.ewm(span=period, min_periods=period, adjust=False).mean()
    rs = avg_gain / avg_loss.replace(0, np.inf)
    rsi = 100 - (100 / (1 + rs))
    return rsi.values

def calculate_adx(high, low, close, period=14):
    """Calculate ADX (Average Directional Index) for trend strength."""
    high_s = pd.Series(high)
    low_s = pd.Series(low)
    close_s = pd.Series(close)
    
    tr1 = high_s - low_s
    tr2 = np.abs(high_s - close_s.shift(1))
    tr3 = np.abs(low_s - close_s.shift(1))
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    
    up_move = high_s - high_s.shift(1)
    down_move = low_s.shift(1) - low_s
    
    plus_dm = up_move.where((up_move > down_move) & (up_move > 0), 0.0)
    minus_dm = down_move.where((down_move > up_move) & (down_move > 0), 0.0)
    
    atr = tr.ewm(span=period, min_periods=period, adjust=False).mean()
    plus_di = 100 * (plus_dm.ewm(span=period, min_periods=period, adjust=False).mean() / atr)
    minus_di = 100 * (minus_dm.ewm(span=period, min_periods=period, adjust=False).mean() / atr)
    
    dx = 100 * np.abs(plus_di - minus_di) / (plus_di + minus_di).replace(0, np.inf)
    adx = dx.ewm(span=period, min_periods=period, adjust=False).mean()
    
    return adx.values

def calculate_ema(close, period=21):
    """Calculate Exponential Moving Average."""
    close_s = pd.Series(close)
    ema = close_s.ewm(span=period, min_periods=period, adjust=False).mean()
    return ema.values

def calculate_volume_ma(volume, period=20):
    """Calculate volume moving average."""
    vol_s = pd.Series(volume)
    vol_ma = vol_s.rolling(window=period, min_periods=period).mean()
    return vol_ma.values

def calculate_bollinger(close, period=20, std_mult=2.0):
    """Calculate Bollinger Bands."""
    close_s = pd.Series(close)
    sma = close_s.rolling(window=period, min_periods=period).mean()
    std = close_s.rolling(window=period, min_periods=period).std()
    upper = sma + std_mult * std
    lower = sma - std_mult * std
    return upper.values, lower.values, sma.values

def generate_signals(prices):
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    volume = prices["volume"].values
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
    
    # Calculate 12h indicators
    atr_14 = calculate_atr(high, low, close, 14)
    adx_14 = calculate_adx(high, low, close, 14)
    rsi_14 = calculate_rsi(close, 14)
    ema_21 = calculate_ema(close, 21)
    ema_8 = calculate_ema(close, 8)
    vol_ma_20 = calculate_volume_ma(volume, 20)
    bb_upper, bb_lower, bb_mid = calculate_bollinger(close, 20, 2.0)
    
    signals = np.zeros(n)
    
    # Position sizing - discrete levels (Rule 4)
    SIZE = 0.28
    
    # Track position state for stoploss
    in_position = False
    position_side = 0
    highest_close = 0.0
    lowest_close = 0.0
    entry_price = 0.0
    
    for i in range(100, n):
        # Skip if indicators not ready
        if np.isnan(atr_14[i]) or atr_14[i] == 0:
            signals[i] = 0.0
            continue
        
        if np.isnan(hma_1d_aligned[i]) or np.isnan(hma_1w_aligned[i]):
            signals[i] = 0.0
            continue
        
        if np.isnan(adx_14[i]) or np.isnan(rsi_14[i]):
            signals[i] = 0.0
            continue
        
        if np.isnan(ema_21[i]) or np.isnan(ema_8[i]) or np.isnan(vol_ma_20[i]):
            signals[i] = 0.0
            continue
        
        # === REGIME DETECTION (Dual HTF HMA) ===
        # Bull regime: price > 1d HMA > 1w HMA
        bull_regime = (close[i] > hma_1d_aligned[i]) and (hma_1d_aligned[i] > hma_1w_aligned[i])
        # Bear regime: price < 1d HMA < 1w HMA
        bear_regime = (close[i] < hma_1d_aligned[i]) and (hma_1d_aligned[i] < hma_1w_aligned[i])
        # Neutral/mixed: everything else
        neutral_regime = not bull_regime and not bear_regime
        
        # === TREND STRENGTH (loose threshold for trade generation) ===
        trend_strong = adx_14[i] > 15  # Loose: ADX>15 not >25
        trend_very_strong = adx_14[i] > 25
        
        # === VOLUME CONFIRMATION ===
        vol_confirmed = volume[i] > 1.2 * vol_ma_20[i] if not np.isnan(vol_ma_20[i]) else False
        
        # === RSI PULLBACK LEVELS ===
        rsi_oversold = rsi_14[i] < 40
        rsi_overbought = rsi_14[i] > 60
        rsi_extreme_low = rsi_14[i] < 30
        rsi_extreme_high = rsi_14[i] > 70
        
        # === EMA CROSSOVER ===
        ema_bull_cross = ema_8[i] > ema_21[i]
        ema_bear_cross = ema_8[i] < ema_21[i]
        
        # === BOLLINGER BAND POSITION ===
        near_bb_lower = close[i] < bb_lower[i] * 1.005 if not np.isnan(bb_lower[i]) else False
        near_bb_upper = close[i] > bb_upper[i] * 0.995 if not np.isnan(bb_upper[i]) else False
        
        # === ENTRY LOGIC (Asymmetric by regime) ===
        new_signal = 0.0
        
        # BULL REGIME: Only long entries
        if bull_regime:
            # Primary: EMA crossover + RSI pullback + volume
            if ema_bull_cross and rsi_oversold and vol_confirmed:
                new_signal = SIZE
            # Secondary: RSI extreme in strong trend
            elif trend_strong and rsi_extreme_low:
                new_signal = SIZE
            # Tertiary: BB lower touch in uptrend
            elif near_bb_lower and ema_bull_cross:
                new_signal = SIZE
        
        # BEAR REGIME: Only short entries
        elif bear_regime:
            # Primary: EMA crossover + RSI pullback + volume
            if ema_bear_cross and rsi_overbought and vol_confirmed:
                new_signal = -SIZE
            # Secondary: RSI extreme in strong trend
            elif trend_strong and rsi_extreme_high:
                new_signal = -SIZE
            # Tertiary: BB upper touch in downtrend
            elif near_bb_upper and ema_bear_cross:
                new_signal = -SIZE
        
        # NEUTRAL REGIME: Mean reversion at BB extremes
        else:
            # Long at BB lower with RSI confirmation
            if near_bb_lower and rsi_extreme_low:
                new_signal = SIZE * 0.5  # Half size in neutral
            # Short at BB upper with RSI confirmation
            elif near_bb_upper and rsi_extreme_high:
                new_signal = -SIZE * 0.5  # Half size in neutral
        
        # === STOPLOSS LOGIC (Rule 6) - 2.5 * ATR trailing ===
        if in_position and position_side != 0:
            if position_side > 0:
                # Update highest close for long position
                if close[i] > highest_close:
                    highest_close = close[i]
                stoploss_price = highest_close - 2.5 * atr_14[i]
                if close[i] < stoploss_price:
                    new_signal = 0.0
            
            if position_side < 0:
                # Update lowest close for short position
                if lowest_close == 0.0 or close[i] < lowest_close:
                    lowest_close = close[i]
                stoploss_price = lowest_close + 2.5 * atr_14[i]
                if close[i] > stoploss_price:
                    new_signal = 0.0
        
        # === REGIME REVERSAL EXIT ===
        # Exit if regime flips against position
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