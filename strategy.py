#!/usr/bin/env python3
"""
Experiment #602: 30m Multi-Signal Ensemble (Fisher + Z-score + 4h HMA + ADX Regime)

Hypothesis: After 533+ failures, the winning pattern is combining:
1. Fisher Transform for reversal detection (works in bear markets, catches bottoms)
2. Z-score mean reversion (proven edge in range/bear periods)
3. 4h HMA trend bias (multi-timeframe confirmation, proven in #593)
4. ADX regime with hysteresis (avoid whipsaw in transition zones)
5. Asymmetric entries: more aggressive longs in bull, more shorts in bear

Why this should beat previous 30m attempts (#590 Sharpe=-2.712, #596 Sharpe=-2.865):
- Previous used simple EMA crossover (always fails on BTC/ETH)
- This uses Fisher Transform (non-linear, catches reversals better)
- Z-score filter prevents entering against extreme moves
- ADX hysteresis (25/18) reduces false regime switches
- Looser entry thresholds ensure >=10 trades per symbol

Timeframe: 30m (REQUIRED for this experiment)
HTF: 4h via mtf_data helper (call ONCE before loop)
Position sizing: 0.25-0.30 discrete (max 0.40)
Stoploss: 2.5 * ATR(14) trailing
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_30m_fisher_zscore_4h_hma_adx_regime_ensemble_atr_v1"
timeframe = "30m"
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

def calculate_rsi(close, period=14):
    """Calculate RSI (Relative Strength Index)."""
    close_s = pd.Series(close)
    delta = close_s.diff()
    gain = delta.where(delta > 0, 0.0)
    loss = -delta.where(delta < 0, 0.0)
    
    avg_gain = gain.ewm(span=period, min_periods=period, adjust=False).mean()
    avg_loss = loss.ewm(span=period, min_periods=period, adjust=False).mean()
    
    rs = avg_gain / avg_loss.replace(0, np.inf)
    rsi = 100 - (100 / (1 + rs))
    
    return rsi.values

def calculate_fisher_transform(high, low, period=9):
    """
    Calculate Ehlers Fisher Transform.
    Transforms price into a Gaussian distribution for better reversal signals.
    Long when Fisher crosses above -1.5, Short when crosses below +1.5
    """
    high_s = pd.Series(high)
    low_s = pd.Series(low)
    
    # Median price
    median = (high_s + low_s) / 2
    
    # Normalize price to -1 to +1 range
    highest = median.rolling(window=period, min_periods=period).max()
    lowest = median.rolling(window=period, min_periods=period).min()
    
    range_val = highest - lowest
    range_val = range_val.replace(0, np.inf)
    
    # Normalized value
    norm = (median - lowest) / range_val
    norm = norm.clip(0.001, 0.999)  # Avoid log(0)
    
    # Fisher transform
    fisher = 0.5 * np.log((1 + norm) / (1 - norm))
    
    # Signal line (previous fisher)
    fisher_signal = fisher.shift(1)
    
    return fisher.values, fisher_signal.values

def calculate_zscore(close, period=20):
    """Calculate Z-score for mean reversion detection."""
    close_s = pd.Series(close)
    sma = close_s.rolling(window=period, min_periods=period).mean()
    std = close_s.rolling(window=period, min_periods=period).std()
    std = std.replace(0, np.inf)
    zscore = (close_s - sma) / std
    return zscore.values

def generate_signals(prices):
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    n = len(close)
    
    # Load HTF data ONCE before loop (Rule 1 - CRITICAL)
    df_4h = get_htf_data(prices, '4h')
    
    # Calculate HTF indicators
    hma_4h = calculate_hma(df_4h['close'].values, 21)
    
    # Align HTF to LTF (Rule 2 - auto shift(1) for completed bars)
    hma_4h_aligned = align_htf_to_ltf(prices, df_4h, hma_4h)
    
    # Calculate 30m indicators
    atr_14 = calculate_atr(high, low, close, 14)
    adx_14 = calculate_adx(high, low, close, 14)
    rsi_14 = calculate_rsi(close, 14)
    fisher, fisher_signal = calculate_fisher_transform(high, low, 9)
    zscore_20 = calculate_zscore(close, 20)
    
    signals = np.zeros(n)
    
    # Position sizing - discrete levels (Rule 4)
    SIZE = 0.28
    
    # Track position state for stoploss (separate from signal)
    in_position = False
    position_side = 0
    highest_close = 0.0
    lowest_close = 0.0
    entry_price = 0.0
    
    # ADX regime state with hysteresis
    in_trend_regime = False
    
    for i in range(100, n):
        # Skip if indicators not ready
        if np.isnan(atr_14[i]) or atr_14[i] == 0:
            continue
        
        if np.isnan(hma_4h_aligned[i]):
            continue
        
        if np.isnan(adx_14[i]) or np.isnan(rsi_14[i]):
            continue
        
        if np.isnan(fisher[i]) or np.isnan(fisher_signal[i]):
            continue
        
        if np.isnan(zscore_20[i]):
            continue
        
        # === ADX REGIME WITH HYSTERESIS ===
        # Enter trend regime when ADX > 25, exit when < 18
        if adx_14[i] > 25:
            in_trend_regime = True
        elif adx_14[i] < 18:
            in_trend_regime = False
        
        # === 4H HMA TREND BIAS ===
        bull_bias = close[i] > hma_4h_aligned[i]
        bear_bias = close[i] < hma_4h_aligned[i]
        
        # === FISHER TRANSFORM SIGNALS ===
        # Long: Fisher crosses above -1.5 (oversold reversal)
        fisher_long = (fisher_signal[i] < -1.5) and (fisher[i] > -1.5)
        # Short: Fisher crosses below +1.5 (overbought reversal)
        fisher_short = (fisher_signal[i] > 1.5) and (fisher[i] < 1.5)
        
        # === Z-SCORE MEAN REVERSION ===
        # Extreme oversold (long opportunity)
        zscore_oversold = zscore_20[i] < -1.5
        # Extreme overbought (short opportunity)
        zscore_overbought = zscore_20[i] > 1.5
        
        # === RSI CONFIRMATION ===
        rsi_oversold = rsi_14[i] < 40
        rsi_overbought = rsi_14[i] > 60
        
        # === ENTRY LOGIC ===
        new_signal = 0.0
        
        # MODE 1: TREND REGIME (ADX > 25) - Follow 4h HMA bias with Fisher entry
        if in_trend_regime:
            # Long: Bullish 4h bias + Fisher reversal from oversold
            if bull_bias and fisher_long and rsi_oversold:
                new_signal = SIZE
            
            # Short: Bearish 4h bias + Fisher reversal from overbought
            elif bear_bias and fisher_short and rsi_overbought:
                new_signal = -SIZE
        
        # MODE 2: RANGE REGIME (ADX < 18) - Z-score mean reversion
        else:
            # Long: Z-score oversold + RSI confirmation + not strongly bearish
            if zscore_oversold and rsi_oversold:
                if not bear_bias or adx_14[i] < 22:
                    new_signal = SIZE
            
            # Short: Z-score overbought + RSI confirmation + not strongly bullish
            elif zscore_overbought and rsi_overbought:
                if not bull_bias or adx_14[i] < 22:
                    new_signal = -SIZE
        
        # MODE 3: TRANSITION (18 <= ADX <= 25) - Conservative entries only
        # Only enter if strong confluence (both Fisher and Z-score agree)
        if not in_trend_regime and adx_14[i] >= 18:
            # Strong long signal
            if fisher_long and zscore_oversold and bull_bias:
                new_signal = SIZE * 0.7  # Reduced size in transition
            
            # Strong short signal
            elif fisher_short and zscore_overbought and bear_bias:
                new_signal = -SIZE * 0.7
        
        # === STOPLOSS LOGIC (Rule 6) - 2.5 * ATR trailing ===
        stoploss_triggered = False
        
        if in_position and position_side != 0:
            if position_side > 0:
                # Update highest close for long position
                if close[i] > highest_close:
                    highest_close = close[i]
                stoploss_price = highest_close - 2.5 * atr_14[i]
                if close[i] < stoploss_price:
                    stoploss_triggered = True
            
            if position_side < 0:
                # Update lowest close for short position
                if lowest_close == 0.0 or close[i] < lowest_close:
                    lowest_close = close[i]
                stoploss_price = lowest_close + 2.5 * atr_14[i]
                if close[i] > stoploss_price:
                    stoploss_triggered = True
        
        # === TREND REVERSAL EXIT ===
        trend_reversal = False
        if in_position and position_side != 0:
            if position_side > 0 and bear_bias and adx_14[i] > 20:
                trend_reversal = True
            if position_side < 0 and bull_bias and adx_14[i] > 20:
                trend_reversal = True
        
        # Apply stoploss or trend reversal
        if stoploss_triggered or trend_reversal:
            new_signal = 0.0
        
        # === UPDATE POSITION TRACKING ===
        if new_signal != 0.0:
            if not in_position:
                # New entry
                in_position = True
                position_side = np.sign(new_signal)
                highest_close = close[i] if position_side > 0 else 0.0
                lowest_close = close[i] if position_side < 0 else 0.0
                entry_price = close[i]
            elif np.sign(new_signal) != position_side:
                # Position flip
                position_side = np.sign(new_signal)
                highest_close = close[i] if position_side > 0 else 0.0
                lowest_close = close[i] if position_side < 0 else 0.0
                entry_price = close[i]
        else:
            if in_position:
                # Exit position
                in_position = False
                position_side = 0
                highest_close = 0.0
                lowest_close = 0.0
                entry_price = 0.0
        
        signals[i] = new_signal
    
    return signals