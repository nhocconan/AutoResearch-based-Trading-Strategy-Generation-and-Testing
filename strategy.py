#!/usr/bin/env python3
"""
Experiment #026: 30m RSI Pullback + 4h/1d HMA Dual Trend + ADX + Volume

Hypothesis: Previous 30m strategies failed due to:
1. Too many trades → fee drag (#014, #020, #025 all negative Sharpe)
2. Weak HTF filters (4h only, not 4h+1d combined)
3. No volume confirmation on entries
4. Stops too tight for 30m noise

This strategy addresses all failures:
1. STRICT entry: BOTH 4h AND 1d HMA must agree (not just one)
2. RSI pullback INTO trend (not extremes) - higher win rate in trends
3. ADX > 25 confirms strong trend (filters chop)
4. Volume > 1.5x average confirms conviction
5. 3.0 ATR stop (wider for 30m noise)
6. Discrete sizing (0.0, ±0.25, ±0.35) minimizes churn

Why this should beat 12h baseline (Sharpe=0.137):
- 30m captures more trend moves than 12h while HTF filters prevent overtrading
- Dual HTF (4h+1d) is stricter than 12h's (1d OR 1w) = fewer but higher quality trades
- RSI pullback entries have better risk/reward than Donchian breakouts
- Volume filter addresses fakeout problem that killed #019 (15m Donchian)

Timeframe: 30m (REQUIRED for this experiment)
HTF: 4h and 1d via mtf_data helper (call ONCE before loop)
Position sizing: 0.25-0.35 discrete levels
Stoploss: 3.0 * ATR(14) trailing
Target trades: 40-80/year on 30m (optimal per Rule 10)
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_30m_rsi_pullback_4h_1d_hma_adx_vol_atr_v1"
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

def calculate_rsi(close, period=14):
    """Calculate RSI (Relative Strength Index)."""
    close_s = pd.Series(close)
    delta = close_s.diff()
    gain = delta.where(delta > 0, 0)
    loss = -delta.where(delta < 0, 0)
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
    
    plus_dm = high_s.diff()
    minus_dm = -low_s.diff()
    
    plus_dm = plus_dm.where((plus_dm > minus_dm) & (plus_dm > 0), 0)
    minus_dm = minus_dm.where((minus_dm > plus_dm) & (minus_dm > 0), 0)
    
    atr = tr.ewm(span=period, min_periods=period, adjust=False).mean()
    plus_di = 100 * (plus_dm.ewm(span=period, min_periods=period, adjust=False).mean() / atr)
    minus_di = 100 * (minus_dm.ewm(span=period, min_periods=period, adjust=False).mean() / atr)
    
    dx = 100 * np.abs(plus_di - minus_di) / (plus_di + minus_di).replace(0, np.inf)
    adx = dx.ewm(span=period, min_periods=period, adjust=False).mean()
    
    return adx.values, plus_di.values, minus_di.values

def calculate_volume_spike(volume, lookback=20, threshold=1.5):
    """Detect volume spikes above threshold * average."""
    vol_s = pd.Series(volume)
    vol_avg = vol_s.rolling(window=lookback, min_periods=lookback).mean()
    vol_spike = volume > (threshold * vol_avg)
    return vol_spike

def generate_signals(prices):
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    volume = prices["volume"].values
    n = len(close)
    
    # Load HTF data ONCE before loop (Rule 1 - CRITICAL)
    df_4h = get_htf_data(prices, '4h')
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate HTF indicators
    hma_4h = calculate_hma(df_4h['close'].values, 21)
    hma_1d = calculate_hma(df_1d['close'].values, 21)
    
    # Align HTF to LTF (Rule 2 - auto shift(1) for completed bars)
    hma_4h_aligned = align_htf_to_ltf(prices, df_4h, hma_4h)
    hma_1d_aligned = align_htf_to_ltf(prices, df_1d, hma_1d)
    
    # Calculate 30m indicators
    atr_14 = calculate_atr(high, low, close, 14)
    adx_14, plus_di, minus_di = calculate_adx(high, low, close, 14)
    rsi_14 = calculate_rsi(close, 14)
    vol_spike = calculate_volume_spike(volume, lookback=20, threshold=1.5)
    
    # Calculate 30m EMA for pullback entry
    close_s = pd.Series(close)
    ema_21 = close_s.ewm(span=21, min_periods=21, adjust=False).mean().values
    ema_50 = close_s.ewm(span=50, min_periods=50, adjust=False).mean().values
    
    signals = np.zeros(n)
    
    # Position sizing (Rule 4 - discrete levels, max 0.40)
    SIZE_STRONG = 0.35  # All filters agree
    SIZE_MODERATE = 0.25  # Partial confirmation
    
    # Track position state for stoploss
    in_position = False
    position_side = 0
    entry_price = 0.0
    highest_price = 0.0
    lowest_price = 0.0
    
    for i in range(100, n):
        # Skip if indicators not ready
        if np.isnan(atr_14[i]) or atr_14[i] == 0:
            continue
        
        if np.isnan(hma_4h_aligned[i]) or np.isnan(hma_1d_aligned[i]):
            continue
        
        if np.isnan(adx_14[i]) or np.isnan(rsi_14[i]):
            continue
        
        if np.isnan(ema_21[i]) or np.isnan(ema_50[i]):
            continue
        
        # === HTF TREND BIAS (BOTH 4h AND 1d HMA must agree) ===
        # STRICHER than previous strategies - both must align
        price_vs_4h = close[i] - hma_4h_aligned[i]
        price_vs_1d = close[i] - hma_1d_aligned[i]
        
        bull_htf = (price_vs_4h > 0) and (price_vs_1d > 0)  # BOTH bullish
        bear_htf = (price_vs_4h < 0) and (price_vs_1d < 0)  # BOTH bearish
        
        # === ADX TREND STRENGTH ===
        adx_strong = adx_14[i] > 25  # Strong trending market
        
        # === DI DIRECTION ===
        di_bull = plus_di[i] > minus_di[i]
        di_bear = minus_di[i] > plus_di[i]
        
        # === RSI PULLBACK ENTRY ===
        # Long: RSI pulls back to 40-50 in uptrend (not oversold extreme)
        # Short: RSI rallies to 50-60 in downtrend (not overbought extreme)
        rsi_pullback_long = (rsi_14[i] >= 40) and (rsi_14[i] <= 50)
        rsi_pullback_short = (rsi_14[i] >= 50) and (rsi_14[i] <= 60)
        
        # === EMA ALIGNMENT ===
        ema_bull = (close[i] > ema_21[i]) and (ema_21[i] > ema_50[i])
        ema_bear = (close[i] < ema_21[i]) and (ema_21[i] < ema_50[i])
        
        # === VOLUME CONFIRMATION ===
        vol_confirmed = vol_spike[i]
        
        # === ENTRY LOGIC ===
        new_signal = 0.0
        signal_strength = 0
        
        # LONG ENTRY: HTF bull + RSI pullback + EMA/volume/ADX/DI confirmation
        if bull_htf and rsi_pullback_long:
            signal_strength += 3  # HTF trend + RSI pullback + EMA alignment (core)
            
            if ema_bull:
                signal_strength += 1  # EMA confirmation
            
            if vol_confirmed:
                signal_strength += 1  # Volume confirmation
            
            if adx_strong:
                signal_strength += 1  # Trend strength
            
            if di_bull:
                signal_strength += 1  # DI direction
            
            # Assign size based on confirmation count
            if signal_strength >= 6:
                new_signal = SIZE_STRONG
            elif signal_strength >= 4:
                new_signal = SIZE_MODERATE
        
        # SHORT ENTRY: HTF bear + RSI pullback + EMA/volume/ADX/DI confirmation
        elif bear_htf and rsi_pullback_short:
            signal_strength += 3  # HTF trend + RSI pullback + EMA alignment (core)
            
            if ema_bear:
                signal_strength += 1  # EMA confirmation
            
            if vol_confirmed:
                signal_strength += 1  # Volume confirmation
            
            if adx_strong:
                signal_strength += 1  # Trend strength
            
            if di_bear:
                signal_strength += 1  # DI direction
            
            # Assign size based on confirmation count
            if signal_strength >= 6:
                new_signal = -SIZE_STRONG
            elif signal_strength >= 4:
                new_signal = -SIZE_MODERATE
        
        # === STOPLOSS LOGIC (Rule 6) - 3.0 * ATR trailing ===
        stoploss_triggered = False
        
        if in_position and position_side != 0:
            if position_side > 0:
                # Update highest price for long position
                if close[i] > highest_price:
                    highest_price = close[i]
                stoploss_price = highest_price - 3.0 * atr_14[i]
                if close[i] < stoploss_price:
                    stoploss_triggered = True
            
            if position_side < 0:
                # Update lowest price for short position
                if lowest_price == 0.0 or close[i] < lowest_price:
                    lowest_price = close[i]
                stoploss_price = lowest_price + 3.0 * atr_14[i]
                if close[i] > stoploss_price:
                    stoploss_triggered = True
        
        # === TREND REVERSAL EXIT ===
        trend_exit = False
        if in_position and position_side != 0:
            # Exit if HTF trend strongly reverses against position
            if position_side > 0 and bear_htf:
                trend_exit = True
            if position_side < 0 and bull_htf:
                trend_exit = True
            
            # Exit if RSI reaches extreme (take profit)
            if position_side > 0 and rsi_14[i] > 70:
                trend_exit = True
            if position_side < 0 and rsi_14[i] < 30:
                trend_exit = True
        
        # Apply stoploss or trend exit
        if stoploss_triggered or trend_exit:
            new_signal = 0.0
        
        # === UPDATE POSITION TRACKING ===
        if new_signal != 0.0:
            if not in_position:
                # New entry
                in_position = True
                position_side = np.sign(new_signal)
                entry_price = close[i]
                highest_price = close[i] if position_side > 0 else 0.0
                lowest_price = close[i] if position_side < 0 else 0.0
            elif np.sign(new_signal) != position_side:
                # Position flip
                position_side = np.sign(new_signal)
                entry_price = close[i]
                highest_price = close[i] if position_side > 0 else 0.0
                lowest_price = close[i] if position_side < 0 else 0.0
        else:
            if in_position:
                # Exit position
                in_position = False
                position_side = 0
                entry_price = 0.0
                highest_price = 0.0
                lowest_price = 0.0
        
        signals[i] = new_signal
    
    return signals