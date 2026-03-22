#!/usr/bin/env python3
"""
Experiment #481: 4h Primary + 1d HTF — KAMA Trend + Fisher Transform + ADX Regime

Hypothesis: After 480 experiments, clear patterns emerge for 4h timeframe:
1. KAMA (Kaufman Adaptive MA) outperforms HMA/EMA in mixed regimes - adapts to volatility
2. Fisher Transform catches reversals better than RSI (research: 75% win rate on bear rallies)
3. ADX with hysteresis (enter >25, exit <18) prevents whipsaw in 2022-style crashes
4. 1d KAMA provides cleaner trend bias than HMA for HTF direction
5. Relaxed Fisher thresholds (-1.2/+1.2 instead of -1.5/+1.5) for adequate trade frequency

Why this might beat current best (Sharpe=0.435):
- KAMA adapts to volatility regime automatically (no manual CHOP calculation needed)
- Fisher Transform is proven for bear market reversals (ETH research Sharpe +0.923)
- ADX hysteresis reduces false exits during trend continuation
- 4h has shown promise with KAMA+ADX+RSI (ETH Sharpe +0.755 in research)
- Simpler logic = more trades (critical: need >=30 trades/symbol on train)

Position sizing: 0.25-0.30 (discrete levels, max 0.40)
Stoploss: 2.5 * ATR trailing (signal → 0 when hit)
Target: 20-50 trades/year on 4h, >=30 trades/symbol on train, >=3 on test
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_4h_kama_fisher_adx_1d_regime_v1"
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

def calculate_kama(close, period=10, fast_period=2, slow_period=30):
    """
    Calculate Kaufman Adaptive Moving Average (KAMA).
    KAMA adapts to market volatility - smooth in trends, responsive in ranges.
    ER (Efficiency Ratio) determines smoothing constant.
    """
    n = len(close)
    close_s = pd.Series(close)
    
    # Efficiency Ratio: |net change| / sum of absolute changes
    net_change = np.abs(close_s - close_s.shift(period))
    sum_changes = np.abs(close_s.diff()).rolling(window=period, min_periods=period).sum()
    
    er = net_change / (sum_changes + 1e-10)
    er = er.fillna(0)
    
    # Smoothing constant
    fast_sc = 2.0 / (fast_period + 1)
    slow_sc = 2.0 / (slow_period + 1)
    sc = (er * (fast_sc - slow_sc) + slow_sc) ** 2
    
    # KAMA calculation
    kama = np.zeros(n)
    kama[period] = close[period]
    
    for i in range(period + 1, n):
        kama[i] = kama[i-1] + sc.iloc[i] * (close[i] - kama[i-1])
    
    return kama

def calculate_fisher_transform(high, low, period=9):
    """
    Calculate Ehlers Fisher Transform.
    Transforms price into a Gaussian normal distribution for clearer reversal signals.
    Long when Fisher crosses above -1.5, short when crosses below +1.5.
    """
    n = len(high)
    fisher = np.zeros(n)
    fisher_signal = np.zeros(n)
    
    for i in range(period, n):
        # Calculate typical price
        hl2 = (high[i-period:i+1] + low[i-period:i+1]) / 2.0
        
        # Highest high and lowest low over period
        highest = np.max(high[i-period:i+1])
        lowest = np.min(low[i-period:i+1])
        
        range_hl = highest - lowest
        if range_hl < 1e-10:
            range_hl = 1e-10
        
        # Normalize price to -1 to +1 range
        x = (2.0 * (hl2[-1] - lowest) / range_hl) - 1.0
        x = np.clip(x, -0.999, 0.999)  # Prevent log domain errors
        
        # Fisher transform
        fisher[i] = 0.5 * np.log((1.0 + x) / (1.0 - x))
        
        # Signal line (1-period lag of fisher)
        if i > period:
            fisher_signal[i] = fisher[i-1]
        else:
            fisher_signal[i] = fisher[i]
    
    return fisher, fisher_signal

def calculate_adx(high, low, close, period=14):
    """
    Calculate Average Directional Index (ADX).
    ADX > 25 = trending, ADX < 20 = ranging.
    Uses hysteresis: enter trend mode at 25, exit at 18.
    """
    n = len(close)
    
    high_s = pd.Series(high)
    low_s = pd.Series(low)
    close_s = pd.Series(close)
    
    # True Range
    tr1 = high_s - low_s
    tr2 = np.abs(high_s - close_s.shift(1))
    tr3 = np.abs(low_s - close_s.shift(1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    
    # Directional Movement
    plus_dm = high_s.diff()
    minus_dm = -low_s.diff()
    
    plus_dm = np.where((plus_dm > minus_dm) & (plus_dm > 0), plus_dm, 0)
    minus_dm = np.where((minus_dm > plus_dm) & (minus_dm > 0), minus_dm, 0)
    
    # Smooth with Wilder's method (EMA with span=period)
    tr_smooth = tr.ewm(span=period, min_periods=period, adjust=False).mean()
    plus_dm_smooth = pd.Series(plus_dm).ewm(span=period, min_periods=period, adjust=False).mean()
    minus_dm_smooth = pd.Series(minus_dm).ewm(span=period, min_periods=period, adjust=False).mean()
    
    # Directional Indicators
    plus_di = 100.0 * (plus_dm_smooth / (tr_smooth + 1e-10))
    minus_di = 100.0 * (minus_dm_smooth / (tr_smooth + 1e-10))
    
    # DX and ADX
    dx = 100.0 * np.abs(plus_di - minus_di) / (plus_di + minus_di + 1e-10)
    adx = dx.ewm(span=period, min_periods=period, adjust=False).mean()
    
    return adx.values, plus_di.values, minus_di.values

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

def generate_signals(prices):
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    n = len(close)
    
    # Load 1d HTF data ONCE before loop (Rule 1 - CRITICAL)
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate 1d HTF KAMA (major trend direction)
    kama_1d_21 = calculate_kama(df_1d['close'].values, period=21)
    
    # Align HTF to LTF (Rule 2 - auto shift(1))
    kama_1d_21_aligned = align_htf_to_ltf(prices, df_1d, kama_1d_21)
    
    # Calculate 4h indicators
    atr_14 = calculate_atr(high, low, close, 14)
    kama_4h_10 = calculate_kama(close, period=10)
    kama_4h_30 = calculate_kama(close, period=30)
    fisher, fisher_signal = calculate_fisher_transform(high, low, period=9)
    adx, plus_di, minus_di = calculate_adx(high, low, close, period=14)
    rsi_14 = calculate_rsi(close, 14)
    
    signals = np.zeros(n)
    
    # Position sizing (Rule 4 - discrete, max 0.40)
    LONG_SIZE = 0.30
    SHORT_SIZE = 0.25
    
    # Track position state for stoploss
    in_position = False
    position_side = 0
    entry_price = 0.0
    highest_since_entry = 0.0
    lowest_since_entry = 0.0
    
    # ADX regime tracking with hysteresis
    in_trend_mode = False
    
    for i in range(200, n):
        # Skip if indicators not ready
        if np.isnan(atr_14[i]) or atr_14[i] == 0:
            continue
        if np.isnan(kama_1d_21_aligned[i]):
            continue
        if np.isnan(kama_4h_10[i]) or np.isnan(kama_4h_30[i]):
            continue
        if np.isnan(fisher[i]) or np.isnan(adx[i]) or np.isnan(rsi_14[i]):
            continue
        
        # === 1D MAJOR TREND (KAMA direction) ===
        bull_regime = close[i] > kama_1d_21_aligned[i]
        bear_regime = close[i] < kama_1d_21_aligned[i]
        
        # === ADX REGIME WITH HYSTERESIS ===
        # Enter trend mode at ADX > 25, exit at ADX < 18
        if adx[i] > 25.0:
            in_trend_mode = True
        elif adx[i] < 18.0:
            in_trend_mode = False
        
        # === 4H LOCAL TREND (KAMA crossover) ===
        kama_bullish = kama_4h_10[i] > kama_4h_30[i]
        kama_bearish = kama_4h_10[i] < kama_4h_30[i]
        
        # === FISHER TRANSFORM SIGNALS (relaxed for frequency) ===
        # Long: Fisher crosses above -1.2 from below
        fisher_bull_cross = (fisher[i] > -1.2) and (fisher_signal[i] <= -1.2)
        # Short: Fisher crosses below +1.2 from above
        fisher_bear_cross = (fisher[i] < 1.2) and (fisher_signal[i] >= 1.2)
        
        # Extreme Fisher levels for mean reversion
        fisher_extreme_low = fisher[i] < -1.8
        fisher_extreme_high = fisher[i] > 1.8
        
        # === RSI FILTER ===
        rsi_oversold = rsi_14[i] < 40.0
        rsi_overbought = rsi_14[i] > 60.0
        
        # === ENTRY LOGIC — DUAL REGIME ===
        new_signal = 0.0
        
        # LONG ENTRIES
        if in_trend_mode:
            # Trend-following long: bull regime + KAMA bullish + Fisher confirmation
            if bull_regime and kama_bullish and (fisher[i] > -0.5 or fisher_bull_cross):
                new_signal = LONG_SIZE
            # Pullback long in uptrend
            elif bull_regime and kama_bullish and rsi_oversold and fisher[i] > -1.5:
                new_signal = LONG_SIZE * 0.8
        else:
            # Mean reversion long: extreme Fisher + oversold RSI
            if fisher_extreme_low and rsi_14[i] < 35.0:
                new_signal = LONG_SIZE * 0.7
            # Fisher cross long in ranging market
            elif fisher_bull_cross and rsi_oversold:
                new_signal = LONG_SIZE * 0.6
        
        # SHORT ENTRIES (only if no long signal)
        if new_signal == 0.0:
            if in_trend_mode:
                # Trend-following short: bear regime + KAMA bearish + Fisher confirmation
                if bear_regime and kama_bearish and (fisher[i] < 0.5 or fisher_bear_cross):
                    new_signal = -SHORT_SIZE
                # Pullback short in downtrend
                elif bear_regime and kama_bearish and rsi_overbought and fisher[i] < 1.5:
                    new_signal = -SHORT_SIZE * 0.8
            else:
                # Mean reversion short: extreme Fisher + overbought RSI
                if fisher_extreme_high and rsi_14[i] > 65.0:
                    new_signal = -SHORT_SIZE * 0.7
                # Fisher cross short in ranging market
                elif fisher_bear_cross and rsi_overbought:
                    new_signal = -SHORT_SIZE * 0.6
        
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
        
        # === TAKE PROFIT / EXIT CONDITIONS ===
        # Exit long on extreme Fisher high
        if in_position and position_side > 0 and fisher[i] > 1.5:
            new_signal = 0.0
        # Exit short on extreme Fisher low
        if in_position and position_side < 0 and fisher[i] < -1.5:
            new_signal = 0.0
        
        # Regime flip exit (strong trend reversal)
        if in_position and position_side > 0 and bear_regime and kama_bearish and adx[i] > 20:
            new_signal = 0.0
        if in_position and position_side < 0 and bull_regime and kama_bullish and adx[i] > 20:
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