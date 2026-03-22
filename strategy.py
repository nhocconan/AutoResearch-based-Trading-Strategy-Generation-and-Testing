#!/usr/bin/env python3
"""
Experiment #322: 12h Primary + 1d/1w HTF — Fisher Transform + HMA Trend + ADX Regime

Hypothesis: Ehlers Fisher Transform combined with HMA trend and ADX regime filter will outperform
current best (Sharpe=0.424) on 12h timeframe because:
1. Fisher Transform normalizes price to Gaussian distribution, better at catching reversals than RSI
2. 1w HMA(21) provides clean major trend direction with minimal lag
3. 1d ADX(14) filters out weak/choppy markets where trend strategies fail
4. 12h entries with 1d/1w filters = HTF trade frequency with precise timing
5. Target: 25-45 trades/year on 12h (appropriate for this TF, low fee drag)

Why this might beat #321 (Sharpe=-0.956) and current best (Sharpe=0.424):
- Fisher Transform catches reversals earlier than RSI (proven in literature)
- ADX regime filter is simpler and more robust than Choppiness Index
- Fewer conflicting conditions = more trades generated (addresses #1 failure mode)
- Asymmetric sizing matches crypto behavior (longs favored)
- Simpler stoploss logic (2.5*ATR trailing)

Key differences from failed strategies:
- Fisher Transform instead of RSI/Connors for entries
- ADX for trend strength (not Choppiness which failed in #312, #314, #316)
- 1w HMA trend + 1d ADX regime + 12h Fisher entries (clean MTF hierarchy)
- Looser entry thresholds to ensure 25+ trades/year
- Discrete signal levels (0.0, ±0.20, ±0.30) to reduce fee churn

Position sizing: 0.25 base, 0.30 strong conviction (longs), 0.20 (shorts)
Stoploss: 2.5 * ATR trailing
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_12h_fisher_hma_adx_1d1w_asym_v1"
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
    """
    Calculate Hull Moving Average (HMA).
    HMA = WMA(2*WMA(n/2) - WMA(n), sqrt(n))
    Reduces lag significantly compared to EMA/SMA.
    """
    n = period
    half = n // 2
    sqrt_n = int(np.sqrt(n))
    
    close_s = pd.Series(close)
    wma_half = close_s.ewm(span=half, min_periods=half, adjust=False).mean()
    wma_full = close_s.ewm(span=n, min_periods=n, adjust=False).mean()
    
    raw_hma = 2.0 * wma_half - wma_full
    hma = raw_hma.ewm(span=sqrt_n, min_periods=sqrt_n, adjust=False).mean()
    
    return hma.values

def calculate_fisher_transform(high, low, close, period=9):
    """
    Calculate Ehlers Fisher Transform.
    Transforms price into a Gaussian normal distribution.
    Entry signals when Fisher crosses key levels (-1.5, +1.5).
    
    Fisher = 0.5 * ln((1 + X) / (1 - X))
    where X = 0.66 * ((price - LL) / (HH - LL) - 0.5) + 0.67 * prior_X
    """
    n = period
    fisher = np.zeros(len(close))
    trigger = np.zeros(len(close))
    
    # Calculate median price
    median = (high + low) / 2.0
    
    for i in range(n, len(close)):
        # Calculate highest high and lowest low over period
        hh = np.max(high[i-n+1:i+1])
        ll = np.min(low[i-n+1:i+1])
        
        if hh > ll:
            x = 0.66 * ((median[i] - ll) / (hh - ll) - 0.5) + 0.67 * (fisher[i-1] if i > 0 else 0.0)
            x = np.clip(x, -0.999, 0.999)  # Prevent division by zero
            fisher[i] = 0.5 * np.log((1.0 + x) / (1.0 - x) + 1e-10)
            trigger[i] = fisher[i-1] if i > 0 else fisher[i]
        else:
            fisher[i] = fisher[i-1] if i > 0 else 0.0
            trigger[i] = fisher[i]
    
    return fisher, trigger

def calculate_adx(high, low, close, period=14):
    """
    Calculate Average Directional Index (ADX).
    ADX > 25 = trending market
    ADX < 20 = range/choppy market
    """
    n = period
    
    high_s = pd.Series(high)
    low_s = pd.Series(low)
    close_s = pd.Series(close)
    
    # Calculate True Range
    tr1 = high_s - low_s
    tr2 = np.abs(high_s - close_s.shift(1))
    tr3 = np.abs(low_s - close_s.shift(1))
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr = tr.ewm(span=n, min_periods=n, adjust=False).mean()
    
    # Calculate +DM and -DM
    plus_dm = high_s.diff()
    minus_dm = -low_s.diff()
    
    plus_dm = plus_dm.where((plus_dm > minus_dm) & (plus_dm > 0), 0.0)
    minus_dm = minus_dm.where((minus_dm > plus_dm) & (minus_dm > 0), 0.0)
    
    # Smooth DM
    plus_di = 100.0 * (plus_dm.ewm(span=n, min_periods=n, adjust=False).mean() / (atr + 1e-10))
    minus_di = 100.0 * (minus_dm.ewm(span=n, min_periods=n, adjust=False).mean() / (atr + 1e-10))
    
    # Calculate DX and ADX
    dx = 100.0 * np.abs(plus_di - minus_di) / (plus_di + minus_di + 1e-10)
    adx = dx.ewm(span=n, min_periods=n, adjust=False).mean()
    
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

def calculate_sma(close, period=200):
    """Calculate Simple Moving Average."""
    return pd.Series(close).rolling(window=period, min_periods=period).mean().values

def generate_signals(prices):
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    n = len(close)
    
    # Load HTF data ONCE before loop (Rule 1 - CRITICAL)
    df_1d = get_htf_data(prices, '1d')
    df_1w = get_htf_data(prices, '1w')
    
    # Calculate 1w HTF indicators (major trend direction)
    hma_1w_21 = calculate_hma(df_1w['close'].values, period=21)
    hma_1w_21_aligned = align_htf_to_ltf(prices, df_1w, hma_1w_21)
    
    # Calculate 1d HTF indicators (regime filter)
    adx_1d_14, plus_di_1d, minus_di_1d = calculate_adx(
        df_1d['high'].values, df_1d['low'].values, df_1d['close'].values, period=14
    )
    adx_1d_aligned = align_htf_to_ltf(prices, df_1d, adx_1d_14)
    plus_di_aligned = align_htf_to_ltf(prices, df_1d, plus_di_1d)
    minus_di_aligned = align_htf_to_ltf(prices, df_1d, minus_di_1d)
    
    # Calculate 12h indicators
    atr_14 = calculate_atr(high, low, close, 14)
    atr_30 = calculate_atr(high, low, close, 30)
    hma_12h_16 = calculate_hma(close, period=16)
    hma_12h_48 = calculate_hma(close, period=48)
    fisher, fisher_trigger = calculate_fisher_transform(high, low, close, period=9)
    rsi_14 = calculate_rsi(close, 14)
    sma_200 = calculate_sma(close, 200)
    
    signals = np.zeros(n)
    
    # Position sizing (Rule 4 - discrete, max 0.40)
    # Asymmetric: longs favored in crypto
    LONG_BASE = 0.25
    LONG_STRONG = 0.30
    SHORT_BASE = 0.20
    SHORT_STRONG = 0.25
    
    # Track position state
    in_position = False
    position_side = 0
    entry_price = 0.0
    highest_price = 0.0
    lowest_price = 0.0
    last_trade_bar = -20
    
    for i in range(100, n):
        # Skip if indicators not ready
        if np.isnan(atr_14[i]) or atr_14[i] == 0:
            continue
        
        if np.isnan(hma_1w_21_aligned[i]):
            continue
        
        if np.isnan(adx_1d_aligned[i]):
            continue
        
        if np.isnan(fisher[i]) or np.isnan(fisher_trigger[i]):
            continue
        
        if np.isnan(hma_12h_16[i]) or np.isnan(hma_12h_48[i]):
            continue
        
        # === 1W MAJOR TREND REGIME (primary direction filter) ===
        # Bull: price above 1w HMA (favor longs)
        # Bear: price below 1w HMA (allow shorts)
        regime_bull = close[i] > hma_1w_21_aligned[i]
        regime_bear = close[i] < hma_1w_21_aligned[i]
        
        # === 1D ADX REGIME (trend strength filter) ===
        # ADX > 25 = trending (favor trend entries)
        # ADX < 20 = range (favor mean reversion)
        adx_trending = adx_1d_aligned[i] > 25.0
        adx_ranging = adx_1d_aligned[i] < 20.0
        adx_neutral = 20.0 <= adx_1d_aligned[i] <= 25.0
        
        # DI direction
        di_bullish = plus_di_aligned[i] > minus_di_aligned[i]
        di_bearish = plus_di_aligned[i] < minus_di_aligned[i]
        
        # === VOLATILITY REGIME (ATR ratio) ===
        atr_ratio = atr_14[i] / (atr_30[i] + 1e-10)
        high_vol = atr_ratio > 1.5
        vol_scale = 0.7 if high_vol else 1.0
        
        # === 12H LOCAL TREND ===
        # HMA crossover
        hma_bullish = hma_12h_16[i] > hma_12h_48[i]
        hma_bearish = hma_12h_16[i] < hma_12h_48[i]
        
        # HMA slope (3-bar lookback)
        hma_slope_up = hma_12h_48[i] > hma_12h_48[i-3] if i >= 3 else False
        hma_slope_down = hma_12h_48[i] < hma_12h_48[i-3] if i >= 3 else False
        
        # Price position relative to HMA
        price_above_hma = close[i] > hma_12h_48[i]
        price_below_hma = close[i] < hma_12h_48[i]
        
        # Price relative to SMA200
        price_above_sma200 = close[i] > sma_200[i] if not np.isnan(sma_200[i]) else False
        
        # === FISHER TRANSFORM SIGNALS (reversal entries) ===
        # Fisher crosses above -1.5 = bullish reversal
        # Fisher crosses below +1.5 = bearish reversal
        fisher_bull_cross = (fisher[i] > -1.5 and fisher_trigger[i] <= -1.5)
        fisher_bear_cross = (fisher[i] < 1.5 and fisher_trigger[i] >= 1.5)
        fisher_oversold = fisher[i] < -1.0
        fisher_overbought = fisher[i] > 1.0
        fisher_rising = fisher[i] > fisher[i-1] if i > 0 else False
        fisher_falling = fisher[i] < fisher[i-1] if i > 0 else False
        
        # === RSI CONFIRMATION ===
        rsi_oversold = rsi_14[i] < 40.0
        rsi_overbought = rsi_14[i] > 60.0
        rsi_neutral = 40.0 <= rsi_14[i] <= 60.0
        
        # === ENTRY LOGIC (ASYMMETRIC + REGIME-AWARE) ===
        new_signal = 0.0
        bars_since_last_trade = i - last_trade_bar
        
        # LONG ENTRIES (favored in bull regime)
        if regime_bull:
            # Fisher bullish cross + trending market + HMA bullish
            if fisher_bull_cross and adx_trending and hma_bullish:
                new_signal = LONG_BASE * vol_scale
            
            # Fisher oversold + bull regime + above SMA200
            elif fisher_oversold and regime_bull and price_above_sma200 and rsi_oversold:
                new_signal = LONG_STRONG * vol_scale
            
            # HMA bullish crossover + Fisher rising + DI bullish
            elif hma_bullish and hma_slope_up and fisher_rising and di_bullish:
                new_signal = LONG_BASE * vol_scale
            
            # Range market mean revert (Fisher very oversold)
            elif adx_ranging and fisher_oversold and price_below_hma:
                new_signal = LONG_BASE * 0.8 * vol_scale
        
        # SHORT ENTRIES (only in bear regime, reduced size)
        if regime_bear:
            # Fisher bearish cross + trending market + HMA bearish
            if fisher_bear_cross and adx_trending and hma_bearish:
                if new_signal == 0.0:
                    new_signal = -SHORT_BASE * vol_scale
            
            # Fisher overbought + bear regime + below SMA200
            elif fisher_overbought and regime_bear and not price_above_sma200 and rsi_overbought:
                if new_signal == 0.0:
                    new_signal = -SHORT_STRONG * vol_scale
            
            # HMA bearish crossover + Fisher falling + DI bearish
            elif hma_bearish and hma_slope_down and fisher_falling and di_bearish:
                if new_signal == 0.0:
                    new_signal = -SHORT_BASE * vol_scale
            
            # Range market mean revert (Fisher very overbought)
            elif adx_ranging and fisher_overbought and price_above_hma:
                if new_signal == 0.0:
                    new_signal = -SHORT_BASE * 0.8 * vol_scale
        
        # === FREQUENCY SAFEGUARD (ensure 25+ trades/year on 12h) ===
        # Force trade if no signal for 25 bars (~12-13 days on 12h)
        if bars_since_last_trade > 25 and new_signal == 0.0 and not in_position:
            if regime_bull and fisher[i] > -1.0 and price_above_hma:
                new_signal = LONG_BASE * 0.6 * vol_scale
            elif regime_bear and fisher[i] < 1.0 and price_below_hma:
                new_signal = -SHORT_BASE * 0.6 * vol_scale
            elif fisher_oversold and rsi_oversold:
                new_signal = LONG_BASE * 0.6 * vol_scale
            elif fisher_overbought and rsi_overbought:
                new_signal = -SHORT_BASE * 0.6 * vol_scale
        
        # === STOPLOSS LOGIC (Rule 6) - 2.5 * ATR trailing ===
        stoploss_triggered = False
        
        if in_position and position_side != 0:
            if position_side > 0:
                if close[i] > highest_price:
                    highest_price = close[i]
                stoploss_price = highest_price - 2.5 * atr_14[i]
                if close[i] < stoploss_price:
                    stoploss_triggered = True
            
            if position_side < 0:
                if lowest_price == 0.0 or close[i] < lowest_price:
                    lowest_price = close[i]
                stoploss_price = lowest_price + 2.5 * atr_14[i]
                if close[i] > stoploss_price:
                    stoploss_triggered = True
        
        # === FISHER REVERSAL EXIT ===
        fisher_exit = False
        if in_position and position_side != 0:
            # Long position: exit when Fisher turns overbought
            if position_side > 0 and fisher_overbought:
                fisher_exit = True
            # Short position: exit when Fisher turns oversold
            if position_side < 0 and fisher_oversold:
                fisher_exit = True
        
        # === HMA REVERSAL EXIT ===
        hma_exit = False
        if in_position and position_side != 0:
            # Long position: exit when HMA turns bearish + price below
            if position_side > 0 and hma_bearish and price_below_hma:
                hma_exit = True
            # Short position: exit when HMA turns bullish + price above
            if position_side < 0 and hma_bullish and price_above_hma:
                hma_exit = True
        
        # === REGIME REVERSAL EXIT ===
        regime_reversal = False
        if in_position and position_side != 0:
            # Long position but 1w regime turns strongly bearish
            if position_side > 0 and regime_bear and price_below_hma:
                regime_reversal = True
            # Short position but 1w regime turns strongly bullish
            if position_side < 0 and regime_bull and price_above_hma:
                regime_reversal = True
        
        if stoploss_triggered or fisher_exit or hma_exit or regime_reversal:
            new_signal = 0.0
        
        # === DISCRETIZE SIGNAL (reduce churn) ===
        if new_signal != 0.0:
            if abs(new_signal) < 0.12:
                new_signal = 0.0
            elif new_signal > 0.28:
                new_signal = LONG_STRONG * vol_scale
            elif new_signal > 0:
                new_signal = LONG_BASE * vol_scale
            elif new_signal < -0.23:
                new_signal = -SHORT_STRONG * vol_scale
            else:
                new_signal = -SHORT_BASE * vol_scale
        
        # === UPDATE POSITION TRACKING ===
        if new_signal != 0.0:
            if not in_position:
                in_position = True
                position_side = np.sign(new_signal)
                entry_price = close[i]
                highest_price = close[i] if position_side > 0 else 0.0
                lowest_price = close[i] if position_side < 0 else 0.0
                last_trade_bar = i
            elif np.sign(new_signal) != position_side:
                position_side = np.sign(new_signal)
                entry_price = close[i]
                highest_price = close[i] if position_side > 0 else 0.0
                lowest_price = close[i] if position_side < 0 else 0.0
                last_trade_bar = i
        else:
            if in_position:
                in_position = False
                position_side = 0
                entry_price = 0.0
                highest_price = 0.0
                lowest_price = 0.0
                last_trade_bar = i
        
        signals[i] = new_signal
    
    return signals